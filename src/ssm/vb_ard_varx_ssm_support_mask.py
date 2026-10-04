"""Support-masked Level-1 Hybrid Kalman + VB-ARD estimator.

The candidate mask is structural: diagonal groups are always estimated and an
off-diagonal group is either estimated at every lag or fixed exactly to zero.
Observability scores never enter the posterior update.
"""
import numpy as np

from src.ssm.vb_ard_varx_ssm_bq_controls import HybridVBARDVARXSSMKnownCWithBQControls


class HybridVBARDVARXSSMKnownCWithBQSupportMask(HybridVBARDVARXSSMKnownCWithBQControls):
    """Estimated-B/Q Hybrid VB-ARD with a fixed candidate-edge mask."""

    def __init__(self,*args,candidate_mask,**kwargs):
        super().__init__(*args,**kwargs)
        mask=np.asarray(candidate_mask,dtype=bool)
        if mask.shape!=(self.n_states,self.n_states):
            raise ValueError("candidate_mask must have shape (n_states, n_states).")
        mask=mask.copy();np.fill_diagonal(mask,True)
        self.candidate_mask=mask
        self.fixed_by_mask=~mask
        np.fill_diagonal(self.fixed_by_mask,False)

    def _allowed_columns(self,target):
        sources=np.flatnonzero(self.candidate_mask[target])
        return np.asarray([lag*self.n_states+source for lag in range(self.na) for source in sources],dtype=int)

    def _initial_A(self):
        values=super()._initial_A()
        for matrix in values:matrix[self.fixed_by_mask]=0.
        return values

    def _update_q_A(self,sufficient_statistics):
        G,H=sufficient_statistics["S_zz"],sufficient_statistics["S_zx"]
        dimension=self.na*self.n_states;means=np.zeros((self.n_states,dimension));covariances=[]
        for target in range(self.n_states):
            allowed=self._allowed_columns(target);prior=np.empty(len(allowed))
            for position,column in enumerate(allowed):
                source=column%self.n_states
                prior[position]=self.diagonal_prior_precision if source==target else self.alpha_mean_[target,source]
            precision=G[np.ix_(allowed,allowed)]/self.Q[target,target]+np.diag(prior)
            precision=.5*(precision+precision.T)+self.posterior_jitter*np.eye(len(allowed))
            try:
                chol=np.linalg.cholesky(precision);restricted=np.linalg.solve(chol.T,np.linalg.solve(chol,np.eye(len(allowed))))
            except np.linalg.LinAlgError:restricted=np.linalg.pinv(precision)
            restricted=.5*(restricted+restricted.T)
            if not np.all(np.isfinite(restricted)) or np.any(np.diag(restricted)<=0):
                raise FloatingPointError("Invalid restricted q(A) posterior covariance.")
            means[target,allowed]=restricted@(H[allowed,target]/self.Q[target,target])
            full=np.zeros((dimension,dimension));full[np.ix_(allowed,allowed)]=restricted;covariances.append(full)
        self._beta_means_=means;self.A_row_covariances_=covariances;self.A_mean_matrices_=self._unpack_A(means)
        for matrix in self.A_mean_matrices_:matrix[self.fixed_by_mask]=0.

    def _update_q_alpha(self):
        shape=np.full((self.n_states,self.n_states),np.nan);rate=np.full_like(shape,np.nan);mean=np.full_like(shape,np.nan)
        for target in range(self.n_states):
            for source in range(self.n_states):
                if source==target:continue
                shape[target,source]=self.a0+.5*self.na
                if self.candidate_mask[target,source]:
                    columns=self._source_columns(source);m=self._beta_means_[target,columns]
                    S=self.A_row_covariances_[target][np.ix_(columns,columns)]
                    rate[target,source]=self.b0+.5*(m@m+np.trace(S));mean[target,source]=shape[target,source]/rate[target,source]
                else:
                    # Kept fixed only to satisfy the parent convergence bookkeeping;
                    # this alpha is not used by any masked coefficient update.
                    mean[target,source]=self.a0/self.b0;rate[target,source]=shape[target,source]/mean[target,source]
        offdiag=~np.eye(self.n_states,dtype=bool)
        if np.any(rate[offdiag]<=0) or not np.all(np.isfinite(mean[offdiag])):
            raise FloatingPointError("Invalid masked q(alpha) update.")
        self.alpha_shape_,self.alpha_rate_,self.alpha_mean_=shape,rate,mean;self.alpha_log_mean_=np.log(mean)

    def _refresh_derived(self):
        super()._refresh_derived()
        for lag in range(self.na):
            self.A_variance_matrices_[lag][self.fixed_by_mask]=0.
            self.A_std_matrices_[lag][self.fixed_by_mask]=0.

    def get_diagnostics(self):
        out=super().get_diagnostics();out.update({"support_masked":True,
            "number_candidate_offdiag_edges":int(np.sum(self.candidate_mask)-self.n_states),
            "candidate_offdiag_density":float((np.sum(self.candidate_mask)-self.n_states)/(self.n_states*(self.n_states-1)))})
        return out


__all__=["HybridVBARDVARXSSMKnownCWithBQSupportMask"]
