import torch
from torch.autograd import Function

class UniversalAttention(Function):
    @staticmethod
    def forward(kc, vc, xq, static_src, static_dest):
        '''
        Inputs:
        kc: b h n_ c_ d
        vc: b h n_ c_ d
        xq: b h r _n _c d
        static_src: b h n_ c_
        static_dest: b h _n _c

        Outputs:
        out: b h r l d n_
        denom: b h r l n_

        Intermediate: 
        variables_ are split over cols
        _variables are split over rows
        (i.e. n_ is the number of col chunks, _n is the number of row chunks)
        '''
        _test = 1
        
        b,h,r,_n,_c,d = xq.shape
        _,_,n_,c_,_ = kc.shape
        l = n_ * c_
        mask = torch.ones(c_,_c, dtype=torch.bool, device=xq.device)
        out = torch.empty(b,h,r,l,d,n, dtype=xq.dtype, device=xq.device)
        denom = torch.empty(b,h,r,l,n, dtype=xq.dtype, device=xq.device)
        static_src = static_src.pow(1/3)
        static_dest = static_dest.pow(1/3)
        kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c
        
        # Iteration over columns
        for i in range(n_):
            k_ = kc[:,:,i]  # b h c_ d
            v_ = vc[:,:,i]  # b h c_ d
            static_src_ = static_src[:,:,i]  # b h c_
            sum_buffer = torch.zeros(b,h,c_, dtype=static_src.dtype, device=static_src.device)

            # Iteration over rows
            for j in range(_n):
                _q = xq[:,:,:,j]  # b h r _c d
                _static_dest = static_dest[:,:,j]  # b h _c
                _kt = kt[:,:,:,j]  # b h d _c

                # Calculate decay matrix
                affinity = k_.matmul(_kt).relu().pow(2/3).float()  # deltanet style decay: b h c_ _c
                affinity = affinity * static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)  # incorporate mamba-style and per-token decay
                affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c_ _c
                affinity = affinity.triu(i*c_-j*_c+1).cumsum(3)  # Accumulate decay with causal masking, within row chunk
                affinity = affinity + sum_buffer.unsqueeze(-1)  # Accumulate decay from prior row chunk
                sum_buffer = affinity[:,:,:,-1]
                affinity = affinity.masked_fill(mask.tril(i*c_-j*_c-1), -1e12)  # Re-mask, with 1s on diagonal

                # Perform actual attention operation
                score = k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))  # b h r c_ _c
                _denom_ = score.logsumexp(dim=-2)  # b h r _c
                _out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))  # b h r _c d

                out[:,:,:,j*_c:(j+1)*_c,:,i] = _out_
                denom[:,:,:,j*_c:(j+1)*_c,i] = _denom_
        return out, denom

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        kc,vc,xq,ss,sd = inputs
        # out,denom = outputs
        ctx.save_for_backward(kc,vc,xq,ss,sd)

    @staticmethod
    def backward(ctx, g_out, g_denom):
        '''
        Inputs:
        kc: b h n_ c_ d
        vc: b h n_ c_ d
        xq: b h r _n _c d
        static_src: b h n_ c_
        static_dest: b h _n _c

        Outputs:
        out: b h r l d n_
        denom: b h r l n_

        Intermediate: 
        variables_ are split over cols
        _variables are split over rows
        (i.e. n_ is the number of col chunks, _n is the number of row chunks)

        Note: when using mixed precision, g_out is downcast but g_denom is always fp32
        '''
        kc,vc,xq,static_src,static_dest = ctx.saved_tensors
        dkc,dvc,dxq,dstat_src,dstat_dest = [torch.zeros_like(x) for x in [kc,vc,xq,static_src,static_dest]]
        b,h,r,_n,_c,d = xq.shape
        _,_,n_,c_,_ = kc.shape
        l = n_ * c_
        mask = torch.ones(c_,_c, dtype=torch.bool, device=xq.device)
        static_src = static_src.pow(1/3)
        static_dest = static_dest.pow(1/3)
        kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c

        # Iterate over columns
        sum_buffer = torch.empty(b,h,c_, dtype=static_src.dtype, device=static_src.device)
        for i in range(n_):
            k_ = kc[:,:,i]  # b h c_ d
            v_ = vc[:,:,i]  # b h c_ d
            static_src_ = static_src[:,:,i]  # b h c_
            dout_ = g_out[...,i]  # b h r l d
            ddenom_ = g_denom[...,i]  # b h r l

            # Rerun forward pass
            aff1 = torch.empty(b,h,c_,l, dtype=k_.dtype, device=k_.device)
            sscore = torch.empty(b,h,r,c_,l, dtype=torch.float, device=k_.device)
            sum_buffer.zero_()
            # Iterate over rows
            for j in range(_n):
                _q = xq[:,:,:,j]  # b h r _c d
                _static_dest = static_dest[:,:,j]  # b h _c
                _kt = kt[:,:,:,j]  # b h d _c
                affinity = k_.matmul(_kt)  # b h c_ _c
                aff1[:,:,:,j*_c:(j+1)*_c] = affinity
                affinity = affinity.relu().pow(2/3).float() * static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)
                affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c_ _c
                affinity = affinity.triu(i*c_-j*_c+1).cumsum(3)  # Accumulate decay with causal masking, within row chunk
                affinity += sum_buffer.unsqueeze(-1)  # Accumulate decay from prior row chunk
                sum_buffer = affinity[:,:,:,-1]
                affinity = affinity.masked_fill(mask.tril(i*c_-j*_c-1), -1e12)  # Re-mask, with 1s on diagonal
                sscore[:,:,:,:,j*_c:(j+1)*_c] = k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2)).softmax(dim=-2)  # b h r c_ _c

                
            # Backward pass
            sum_buffer.zero_()
            # Iterate over rows, backward
            for j in range(_n-1,-1,-1):
                _q = xq[:,:,:,j]  # b h r _c d
                _static_dest = static_dest[:,:,j]  # b h _c
                _kt = kt[:,:,:,j]  # b h d _c
                _aff1 = aff1[:,:,:,j*_c:(j+1)*_c]  # b h c_ _c
                _aff2 = _aff1.relu().pow(2/3).float() * static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)
                _sscore = sscore[:,:,:,:,j*_c:(j+1)*_c]  # b h r c_ _c
                _dout_ = dout_[:,:,:,j*_c:(j+1)*_c]  # b h r _c d
                _ddenom_ = ddenom_[:,:,:,j*_c:(j+1)*_c]  # b h r _c

                # Backprop through score/v matmul
                dvc[:,:,i] += _sscore.to(dtype=dvc.dtype).matmul(_dout_).sum(2)  # b h r c_ _c, b h r _c d -> b h c_ d
                _dscore = v_.unsqueeze(2).matmul(_dout_.transpose(-1,-2))  # b h c_ d, b h r _c d -> b h r c_ _c  (from out)
                _dscore = _dscore.sub(_dscore.mul(_sscore).sum(-2,True)).mul(_sscore)  # (from softmax)
                _dscore += _sscore * _ddenom_.unsqueeze(-2)  # (from denom)

                # Backprop through q/k matmul
                dxq[:,:,:,j] += _dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # b h r c_ _c, b h c_ d -> b h r _c d
                dkc[:,:,i] += _dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(_q.flatten(2,3))  # b h r c_ _c, b h r _c d -> b h c_ d

                # Backprop through affinity matrix
                _daff = _dscore.sum(2)  # b h c_ _c
                _daff = _daff.flip([3]).cumsum(3).flip([3])  # (from cumsum)
                _daff += sum_buffer.unsqueeze(-1)  # Accumulate across row chunks
                _daff = _daff.triu(i*c_-j*_c+1)  # Prevent accumulation into masked regions
                sum_buffer = _daff[:,:,:,0].clone()  # Cloning prevents subsequent _aff2 calcs from changing buffer values                
                _daff /= _aff2.clamp(min=1e-6, max=1-1e-6) - 1  # ( from ln(1-x) )
                _daff *= _aff2.le(1-1e-6)
                _dstat = _daff.mul(_aff1.relu().pow(2/3)).to(dtype=static_src.dtype)  # b h c_ _c

                # Backprop into stat_src and stat_dest
                dstat_src[:,:,i] += _dstat.mul(_static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # b h c_ _c, b h _c -> b h c_
                dstat_dest[:,:,j] += _dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(_static_dest.pow(2).mul(3))  # b h c_ _c, b h c_ -> b h _c

                # Backprop into k/k matmul
                _daff *= static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)  # (from prod with statics)
                _daff = _daff.to(dtype=_q.dtype) * _aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(_aff1.gt(0))  # (from relu and pow)
                dkc = dkc.view(b,h,l,d)
                dkc[:,:,j*_c:(j+1)*_c] += _daff.transpose(-1,-2).matmul(k_)  # b h c_ _c, b h c_ d -> b h _c d
                dkc[:,:,i*c_:(i+1)*c_] += _daff.matmul(_kt.transpose(-1,-2))  # b h c_ _c, b h d _c -> b h c_ d
                dkc = dkc.view(b,h,n_,c_,d)

        return dkc,dvc,dxq,dstat_src,dstat_dest


class SMVecMatMul(Function):
    @staticmethod
    def forward(mat, vec):
        # mat: ... d n
        # vec: ... n
        return mat.mul(vec.softmax(dim=-1).unsqueeze(-2)).sum(-1)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        mat, vec = inputs
        ctx.save_for_backward(mat, vec)

    @ staticmethod
    def backward(ctx, g):
        mat, vec = ctx.saved_tensors
        vec = vec.softmax(dim=-1)
        d_mat = g.unsqueeze(-1).mul(vec.unsqueeze(-2))  # ... d n
        d_vec = g.unsqueeze(-1).mul(mat).sum(-2)  # ... n
        d_vec = d_vec.sub(d_vec.mul(vec).sum(-1,True)).mul(vec)
        return d_mat, d_vec
