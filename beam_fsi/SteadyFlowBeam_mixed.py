"""
SteadyFlowBeam_mixed.py
=======================
TensorFlow 2 mixed-form PINN for steady 2D laminar flow past a
cylinder with an attached rigid beam (Turek-Hron CFD benchmark).

This script is the TF2 equivalent of SteadyFlowCylinder_mixed.py
(the TF1 cylinder-only version) but extended to the cylinder+beam
geometry described in:

  Turek, S. & Hron, J. (2006). "Proposal for Numerical Benchmarking
  of Fluid-Structure Interaction between an Elastic Object and
  Laminar Incompressible Flow." Fluid-Structure Interaction,
  Lecture Notes in Computational Science and Engineering.

Physics:
  Mixed-form incompressible Navier-Stokes (steady):
    ρ(u·∇u) = ∇·σ,   σ_ij stored as independent network outputs
    Stream-function formulation: u = ∂ψ/∂y,  v = -∂ψ/∂x

Geometry (Turek-Hron channel):
  Domain  : [0, 1.1] × [0, 0.41]
  Cylinder: center (0.2, 0.2), radius 0.05
  Beam    : length 0.35, thickness 0.02,
            attached at cylinder right edge → x ∈ [0.25, 0.60], y ∈ [0.19, 0.21]

Requirements: tensorflow >= 2.x, numpy, scipy, matplotlib
"""

import numpy as np
import time
import pickle
import scipy.io
import scipy.optimize
import matplotlib.pyplot as plt

import tensorflow as tf

tf.random.set_seed(1234)
np.random.seed(1234)


# ============================================================
# Sampling utilities
# ============================================================

def lhs(n_dim, n_samples, seed=None):
    """Latin Hypercube Sampling in [0, 1]^n_dim.

    Returns array of shape (n_samples, n_dim).
    """
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.random((n_samples, n_dim))
    a, b = cut[:n_samples], cut[1:n_samples + 1]
    rdpoints = a[:, None] + (b - a)[:, None] * u
    H = np.empty_like(rdpoints)
    for j in range(n_dim):
        order = rng.permutation(n_samples)
        H[:, j] = rdpoints[order, j]
    return H


# ============================================================
# Geometry helpers
# ============================================================

def DelObsPT(XY, xc, yc, r, xb0, xb1, yb0, yb1):
    """Remove collocation points inside the cylinder or inside the beam rectangle.

    Parameters
    ----------
    XY   : (N, 2) array of candidate points
    xc, yc, r : cylinder center and radius
    xb0..yb1  : beam bounding box [xb0,xb1] × [yb0,yb1]
    """
    dst     = np.sqrt((XY[:, 0] - xc) ** 2 + (XY[:, 1] - yc) ** 2)
    in_cyl  = dst <= r
    in_beam = ((XY[:, 0] >= xb0) & (XY[:, 0] <= xb1) &
               (XY[:, 1] >= yb0) & (XY[:, 1] <= yb1))
    return XY[~(in_cyl | in_beam), :]


def preprocess_mat(dir_path):
    """Load reference solution from a Fluent / FEniCS .mat file.

    Expected keys: x, y, p, vx, vy.
    Returns (x, y, u, v, p) as (N,1) column arrays.
    """
    data = scipy.io.loadmat(dir_path)
    X, Y = data['x'], data['y']
    P, vx, vy = data['p'], data['vx'], data['vy']
    return (X.flatten()[:, None], Y.flatten()[:, None],
            vx.flatten()[:, None], vy.flatten()[:, None],
            P.flatten()[:, None])


# ============================================================
# Visualisation
# ============================================================

def postProcess(xmin, xmax, ymin, ymax, field_pinn,
                field_ref=None, s=2, alpha=0.5, marker='o'):
    """Scatter-plot u, v, p fields.

    If field_ref is provided, plot PINN (left) vs Reference (right).
    Otherwise plot PINN only (3 panels stacked vertically).
    """
    x_p, y_p, u_p, v_p, p_p = field_pinn

    if field_ref is not None:
        x_r, y_r, u_r, v_r, p_r = field_ref
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(9, 5))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

        def _scat(axh, x, y, c, title, vmin=None, vmax=None):
            cf = axh.scatter(x, y, c=c, alpha=alpha, edgecolors='none',
                             cmap='rainbow', marker=marker, s=int(s),
                             vmin=vmin, vmax=vmax)
            axh.set_aspect('equal'); axh.axis('off')
            axh.set_xlim([xmin, xmax]); axh.set_ylim([ymin, ymax])
            axh.set_title(title, fontsize=9)
            fig.colorbar(cf, ax=axh, fraction=0.046, pad=0.04)

        _scat(ax[0, 0], x_p, y_p, u_p, r'$u$ PINN')
        _scat(ax[1, 0], x_p, y_p, v_p, r'$v$ PINN')
        _scat(ax[2, 0], x_p, y_p, p_p, r'$p$ PINN', vmin=-0.25, vmax=0.40)
        _scat(ax[0, 1], x_r, y_r, u_r, r'$u$ Ref')
        _scat(ax[1, 1], x_r, y_r, v_r, r'$v$ Ref')
        _scat(ax[2, 1], x_r, y_r, p_r, r'$p$ Ref',  vmin=-0.25, vmax=0.40)
    else:
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
        fig.subplots_adjust(hspace=0.3)

        def _scat(axh, x, y, c, title, vmin=None, vmax=None):
            cf = axh.scatter(x, y, c=c, alpha=alpha, edgecolors='none',
                             cmap='jet', marker=marker, s=int(s),
                             vmin=vmin, vmax=vmax)
            axh.set_aspect('equal'); axh.axis('off')
            axh.set_xlim([xmin, xmax]); axh.set_ylim([ymin, ymax])
            axh.set_title(title, fontsize=10)
            fig.colorbar(cf, ax=axh, fraction=0.02, pad=0.01)

        _scat(ax[0], x_p, y_p, u_p, r'$u^*$ (PINN)')
        _scat(ax[1], x_p, y_p, v_p, r'$v^*$ (PINN)')
        _scat(ax[2], x_p, y_p, p_p, r'$p^*$ (PINN)')

    plt.tight_layout()
    plt.show()


def plot_loss_all(model, n_adam=0):
    """Two-panel semilogy plot showing every individual loss component.

    Left panel  : PDE residuals (f_u, f_v, f_s11, f_s22, f_s12, f_p) + total
    Right panel : Boundary conditions (wall, inlet, outlet) + total

    Parameters
    ----------
    model  : trained PINN_beam_flow_TF2 instance
    n_adam : iteration index where Adam ended (draws a dashed vertical line)
    """
    iters = np.arange(len(model.loss_total))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── PDE terms ──────────────────────────────────────────────
    ax1.semilogy(iters, model.loss_total,  'k-',   lw=2.0, label='Total (weighted)')
    ax1.semilogy(iters, model.loss_fu,     'b-',   lw=1.2, label=r'$f_u$ (momentum x)')
    ax1.semilogy(iters, model.loss_fv,     'r-',   lw=1.2, label=r'$f_v$ (momentum y)')
    ax1.semilogy(iters, model.loss_fs11,   'b--',  lw=1.0, label=r'$f_{s_{11}}$ (constitutive)')
    ax1.semilogy(iters, model.loss_fs22,   'r--',  lw=1.0, label=r'$f_{s_{22}}$ (constitutive)')
    ax1.semilogy(iters, model.loss_fs12,   'm--',  lw=1.0, label=r'$f_{s_{12}}$ (shear)')
    ax1.semilogy(iters, model.loss_fp,     'g--',  lw=1.0, label=r'$f_p$ (trace/pressure)')
    if 0 < n_adam < len(iters):
        ax1.axvline(n_adam, color='grey', ls='--', lw=1.2, label='Adam → L-BFGS')
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('Unweighted MSE')
    ax1.set_title('PDE Residuals')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, which='both', ls='--', alpha=0.35)

    # ── BC terms ───────────────────────────────────────────────
    ax2.semilogy(iters, model.loss_total,  'k-',   lw=2.0, label='Total (weighted)')
    ax2.semilogy(iters, model.loss_wall,   'g-',   lw=1.2, label='Wall (no-slip)')
    ax2.semilogy(iters, model.loss_inlet,  'b-',   lw=1.2, label='Inlet (velocity)')
    ax2.semilogy(iters, model.loss_outlet, 'r-',   lw=1.2, label='Outlet (pressure)')
    if 0 < n_adam < len(iters):
        ax2.axvline(n_adam, color='grey', ls='--', lw=1.2, label='Adam → L-BFGS')
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Unweighted MSE')
    ax2.set_title('Boundary Condition Residuals')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, which='both', ls='--', alpha=0.35)

    fig.suptitle('Loss Convergence — Cylinder + Beam PINN', fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# Neural network
# ============================================================

class MLP(tf.keras.Model):
    """Fully-connected network with tanh hidden layers, linear output."""

    def __init__(self, layers_sizes, activation='tanh'):
        super().__init__()
        self.hidden = [
            tf.keras.layers.Dense(w, activation=activation,
                                  kernel_initializer='glorot_normal')
            for w in layers_sizes[:-1]
        ]
        self.out = tf.keras.layers.Dense(
            layers_sizes[-1], activation=None,
            kernel_initializer='glorot_normal'
        )

    def call(self, x, training=False):
        z = x
        for lyr in self.hidden:
            z = lyr(z)
        return self.out(z)


# ============================================================
# PINN class — cylinder + beam geometry
# ============================================================

# Default loss weights — can be overridden per-term via the `weights` argument.
# Keys: w_fu, w_fv, w_fs11, w_fs22, w_fs12, w_fp  (PDE residuals)
#        w_wall, w_inlet, w_outlet                  (boundary conditions)
DEFAULT_WEIGHTS = {
    'w_fu':     1.0,   # x-momentum residual
    'w_fv':     1.0,   # y-momentum residual
    'w_fs11':   1.0,   # constitutive: normal stress 11
    'w_fs22':   1.0,   # constitutive: normal stress 22
    'w_fs12':   1.0,   # constitutive: shear stress 12
    'w_fp':     1.0,   # trace / pressure-recovery constraint
    'w_wall':  10.0,   # no-slip BC (cylinder + beam + channel walls)
    'w_inlet': 10.0,   # prescribed inlet velocity
    'w_outlet': 10.0,  # zero mean pressure at outlet
}


class PINN_beam_flow_TF2:
    """Mixed-form PINN for steady laminar flow past cylinder + rigid beam.

    Network outputs (5 values): ψ, p, s11, s22, s12
      u = ∂ψ/∂y,   v = -∂ψ/∂x
    PDE residuals (Turek-Hron CFD benchmark, steady):
      f_u   = ρ(u u_x + v u_y) - s11_x - s12_y
      f_v   = ρ(u v_x + v v_y) - s12_x - s22_y
      f_s11 = -p + 2μ u_x - s11
      f_s22 = -p + 2μ v_y - s22
      f_s12 = μ(u_y + v_x) - s12
      f_p   = p + ½(s11 + s22)   [trace / pressure recovery]

    Each loss term has its own scalar weight set via the `weights` dict.
    All weights default to DEFAULT_WEIGHTS and can be partially overridden:

        model = PINN_beam_flow_TF2(..., weights={'w_fu': 5.0, 'w_wall': 20.0})
    """

    def __init__(self, Collo, INLET, OUTLET, WALL,
                 uv_layers, lb, ub,
                 rho=1.0, mu=0.02, weights=None):
        self.lb = lb.astype(np.float32).reshape(1, 2)
        self.ub = ub.astype(np.float32).reshape(1, 2)

        self.rho = tf.constant(rho, dtype=tf.float32)
        self.mu  = tf.constant(mu,  dtype=tf.float32)

        # Merge user-supplied weights with defaults
        w = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            w.update(weights)
        self.w = {k: tf.constant(float(v), dtype=tf.float32) for k, v in w.items()}
        self._weights_dict = w   # keep a plain-Python copy for inspection / saving

        self.Collo  = Collo.astype(np.float32)
        self.INLET  = INLET.astype(np.float32)   # columns: x, y, u_ref, v_ref
        self.OUTLET = OUTLET.astype(np.float32)  # columns: x, y
        self.WALL   = WALL.astype(np.float32)    # columns: x, y  (no-slip)

        hidden_widths = uv_layers[1:-1]
        out_dim       = uv_layers[-1]  # must be 5
        self.net = MLP(hidden_widths + [out_dim], activation='tanh')

        # ── Loss history — all lists initialised here, never reset mid-run ──
        # Unweighted raw MSE for each term (useful for diagnosing convergence)
        self.loss_fu     = []   # MSE(f_u)
        self.loss_fv     = []   # MSE(f_v)
        self.loss_fs11   = []   # MSE(f_s11)
        self.loss_fs22   = []   # MSE(f_s22)
        self.loss_fs12   = []   # MSE(f_s12)
        self.loss_fp     = []   # MSE(f_p)
        self.loss_pde    = []   # unweighted sum: Σ MSE(f_i)
        self.loss_wall   = []   # unweighted MSE(u_wall, v_wall)
        self.loss_inlet  = []   # unweighted MSE(u_in − u_ref, v_in − v_ref)
        self.loss_outlet = []   # unweighted (mean p_out)²
        self.loss_total  = []   # weighted total (what the optimiser minimises)

    # ----------------------------------------------------------
    def net_psips(self, x, y):
        """Raw network forward pass.  Returns (ψ, p, s11, s22, s12)."""
        xy  = tf.concat([x, y], axis=1)
        out = self.net(xy)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]

    # ----------------------------------------------------------
    def uv_and_grads(self, x, y):
        """Compute (u, v) and all first-order derivatives needed for the PDE."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            psi, p, s11, s22, s12 = self.net_psips(x, y)
            u =  tape.gradient(psi, y)
            v = -tape.gradient(psi, x)
        u_x  = tape.gradient(u,   x);  u_y  = tape.gradient(u,   y)
        v_x  = tape.gradient(v,   x);  v_y  = tape.gradient(v,   y)
        s11_x = tape.gradient(s11, x)
        s12_x = tape.gradient(s12, x);  s12_y = tape.gradient(s12, y)
        s22_y = tape.gradient(s22, y)
        del tape
        return (u, v, p, s11, s22, s12,
                u_x, u_y, v_x, v_y,
                s11_x, s12_x, s12_y, s22_y)

    # ----------------------------------------------------------
    def residuals(self, x, y):
        """Six PDE residuals for the mixed-form steady N-S equations."""
        rho, mu = self.rho, self.mu
        (u, v, p, s11, s22, s12,
         u_x, u_y, v_x, v_y,
         s11_x, s12_x, s12_y, s22_y) = self.uv_and_grads(x, y)

        f_u   = rho * (u * u_x + v * u_y) - s11_x - s12_y
        f_v   = rho * (u * v_x + v * v_y) - s12_x - s22_y
        f_s11 = -p + 2.0 * mu * u_x - s11
        f_s22 = -p + 2.0 * mu * v_y - s22
        f_s12 = mu * (u_y + v_x) - s12
        f_p   = p + 0.5 * (s11 + s22)
        return f_u, f_v, f_s11, f_s22, f_s12, f_p

    # ----------------------------------------------------------
    def loss_fn(self, Xc, WALL, INLET, OUTLET):
        """Composite loss with per-term weights.

        Returns
        -------
        total : weighted scalar loss (optimiser target)
        lfu, lfv, lfs11, lfs22, lfs12, lfp : unweighted MSE of each PDE residual
        lwall, lin, lout : unweighted MSE of each boundary condition
        """
        mse = lambda t: tf.reduce_mean(tf.square(t))
        w   = self.w

        # ── PDE residuals ──────────────────────────────────────
        xc, yc = Xc[:, 0:1], Xc[:, 1:2]
        fu, fv, fs11, fs22, fs12, fp = self.residuals(xc, yc)
        lfu   = mse(fu);   lfv   = mse(fv)
        lfs11 = mse(fs11); lfs22 = mse(fs22)
        lfs12 = mse(fs12); lfp   = mse(fp)

        loss_pde = (w['w_fu']  * lfu  + w['w_fv']   * lfv  +
                    w['w_fs11']* lfs11+ w['w_fs22']  * lfs22+
                    w['w_fs12']* lfs12+ w['w_fp']    * lfp)

        # ── Wall no-slip ────────────────────────────────────────
        xw, yw = WALL[:, 0:1], WALL[:, 1:2]
        u_w, v_w, *_ = self.uv_and_grads(xw, yw)
        lwall = mse(u_w) + mse(v_w)

        # ── Inlet velocity ──────────────────────────────────────
        xi, yi = INLET[:, 0:1], INLET[:, 1:2]
        ui, vi = INLET[:, 2:3], INLET[:, 3:4]
        u_i, v_i, *_ = self.uv_and_grads(xi, yi)
        lin = mse(u_i - ui) + mse(v_i - vi)

        # ── Outlet pressure ─────────────────────────────────────
        xo, yo = OUTLET[:, 0:1], OUTLET[:, 1:2]
        _, p_o, _, _, _ = self.net_psips(xo, yo)
        lout = tf.square(tf.reduce_mean(p_o))

        total = loss_pde + w['w_wall']*lwall + w['w_inlet']*lin + w['w_outlet']*lout
        return total, lfu, lfv, lfs11, lfs22, lfs12, lfp, lwall, lin, lout

    # ----------------------------------------------------------
    def _to_tf(self):
        """Convert stored numpy arrays to TF tensors once."""
        return (tf.constant(self.Collo,  dtype=tf.float32),
                tf.constant(self.WALL,   dtype=tf.float32),
                tf.constant(self.INLET,  dtype=tf.float32),
                tf.constant(self.OUTLET, dtype=tf.float32))

    def _record(self, total, lfu, lfv, lfs11, lfs22, lfs12, lfp, lwall, lin, lout):
        self.loss_total.append(float(total))
        self.loss_fu.append(float(lfu));     self.loss_fv.append(float(lfv))
        self.loss_fs11.append(float(lfs11)); self.loss_fs22.append(float(lfs22))
        self.loss_fs12.append(float(lfs12)); self.loss_fp.append(float(lfp))
        self.loss_pde.append(float(lfu) + float(lfv) + float(lfs11) +
                             float(lfs22) + float(lfs12) + float(lfp))
        self.loss_wall.append(float(lwall))
        self.loss_inlet.append(float(lin))
        self.loss_outlet.append(float(lout))

    # ----------------------------------------------------------
    def print_weights(self):
        """Print the current loss weights in a readable table."""
        print("Loss weights:")
        print(f"  PDE  — w_fu={self._weights_dict['w_fu']:.2g}  "
              f"w_fv={self._weights_dict['w_fv']:.2g}  "
              f"w_fs11={self._weights_dict['w_fs11']:.2g}  "
              f"w_fs22={self._weights_dict['w_fs22']:.2g}  "
              f"w_fs12={self._weights_dict['w_fs12']:.2g}  "
              f"w_fp={self._weights_dict['w_fp']:.2g}")
        print(f"  BC   — w_wall={self._weights_dict['w_wall']:.2g}  "
              f"w_inlet={self._weights_dict['w_inlet']:.2g}  "
              f"w_outlet={self._weights_dict['w_outlet']:.2g}")

    # ----------------------------------------------------------
    def train_adam(self, iters=10000, lr=5e-4, print_every=200):
        """Adam optimiser loop."""
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        Xc, WALL, INLET, OUTLET = self._to_tf()

        for it in range(iters):
            with tf.GradientTape() as tape:
                ret = self.loss_fn(Xc, WALL, INLET, OUTLET)
            total = ret[0]
            grads = tape.gradient(total, self.net.trainable_variables)
            opt.apply_gradients(zip(grads, self.net.trainable_variables))
            self._record(*[v.numpy() for v in ret])
            if it % print_every == 0:
                lfu, lfv, lfs11, lfs22, lfs12, lfp, lwall, lin, lout = \
                    [v.numpy() for v in ret[1:]]
                print(f"Adam {it:6d}/{iters} | total={total.numpy():.3e} | "
                      f"fu={lfu:.2e} fv={lfv:.2e} "
                      f"fs11={lfs11:.2e} fs22={lfs22:.2e} "
                      f"fs12={lfs12:.2e} fp={lfp:.2e} | "
                      f"wall={lwall:.2e} in={lin:.2e} out={lout:.2e}")

    # ----------------------------------------------------------
    def train_lbfgs(self, maxiter=50000, maxcor=50):
        """L-BFGS-B refinement via scipy.optimize.minimize."""
        Xc, WALL, INLET, OUTLET = self._to_tf()

        def pack():
            return np.concatenate(
                [v.numpy().ravel() for v in self.net.trainable_variables]
            ).astype(np.float64)

        def unpack(flat):
            idx = 0
            for v in self.net.trainable_variables:
                n = v.numpy().size
                v.assign(flat[idx:idx + n].reshape(v.shape).astype(np.float32))
                idx += n

        def loss_and_grad(flat):
            unpack(flat)
            with tf.GradientTape() as tape:
                ret = self.loss_fn(Xc, WALL, INLET, OUTLET)
            total = ret[0]
            grads = tape.gradient(total, self.net.trainable_variables)
            g_flat = np.concatenate(
                [g.numpy().ravel() for g in grads]
            ).astype(np.float64)
            return float(total.numpy()), g_flat

        def callback(xk):
            ret = self.loss_fn(Xc, WALL, INLET, OUTLET)
            self._record(*[v.numpy() for v in ret])
            total, lfu, lfv = ret[0], ret[1], ret[2]
            print(f"  L-BFGS | total={float(total):.3e} "
                  f"fu={float(lfu):.2e} fv={float(lfv):.2e}")

        res = scipy.optimize.minimize(
            fun=loss_and_grad, x0=pack(), jac=True,
            method='L-BFGS-B', callback=callback,
            options={'maxiter': maxiter, 'maxcor': maxcor,
                     'ftol': np.finfo(float).eps}
        )
        unpack(res.x)
        return res

    # ----------------------------------------------------------
    def save(self, path='uvNN_beam_weights'):
        self.net.save_weights(path)

    def load(self, path='uvNN_beam_weights'):
        self.net(tf.zeros((1, 2), dtype=tf.float32))  # build first
        self.net.load_weights(path)

    # ----------------------------------------------------------
    def predict(self, x_star, y_star):
        """Return (u, v, p) numpy arrays."""
        x = tf.constant(x_star.astype(np.float32))
        y = tf.constant(y_star.astype(np.float32))
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            psi, p, _, _, _ = self.net_psips(x, y)
        u =  tape.gradient(psi, y)
        v = -tape.gradient(psi, x)
        del tape
        return u.numpy(), v.numpy(), p.numpy()

    def predict_all(self, x_star, y_star):
        """Return (u, v, p, s11, s22, s12) numpy arrays."""
        x = tf.constant(x_star.astype(np.float32))
        y = tf.constant(y_star.astype(np.float32))
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            psi, p, s11, s22, s12 = self.net_psips(x, y)
        u =  tape.gradient(psi, y)
        v = -tape.gradient(psi, x)
        del tape
        return (u.numpy(), v.numpy(), p.numpy(),
                s11.numpy(), s22.numpy(), s12.numpy())


# ============================================================
# Drag / Lift post-processing
# ============================================================

def _traction(model, x, y, nx, ny):
    """Compute traction vector t = σ n from predicted stress components."""
    x  = np.asarray(x).reshape(-1, 1).astype(np.float32)
    y  = np.asarray(y).reshape(-1, 1).astype(np.float32)
    nx = np.asarray(nx).reshape(-1)
    ny = np.asarray(ny).reshape(-1)
    _, _, _, s11, s22, s12 = model.predict_all(x, y)
    s11 = s11.reshape(-1); s22 = s22.reshape(-1); s12 = s12.reshape(-1)
    return s11 * nx + s12 * ny, s12 * nx + s22 * ny


def compute_drag_lift(model, xc=0.2, yc=0.2, r=0.05,
                      beam_L=0.35, beam_h=0.02,
                      n_theta=2000, n_edge=800,
                      rho=1.0, Ubar=0.2, d=None):
    """Integrate drag and lift forces on the cylinder + beam surface.

    Integration boundaries:
      - Cylinder arc (excluding the beam attachment slot)
      - Beam: top edge, bottom edge, right edge

    Returns
    -------
    D, L : dimensional drag and lift forces [N/m]
    Cd, Cl : non-dimensional coefficients  (based on Ubar and diameter d)
    """
    if d is None:
        d = 2.0 * r

    # --- Cylinder arc (exclude attachment slot) ---
    theta0 = np.arcsin(np.clip(0.5 * beam_h / r, 0.0, 1.0))
    th = np.linspace(theta0, 2 * np.pi - theta0, n_theta, endpoint=False)
    dS_c = r * (2 * np.pi - 2 * theta0) / n_theta
    tx_c, ty_c = _traction(model,
                            xc + r * np.cos(th), yc + r * np.sin(th),
                            np.cos(th), np.sin(th))
    D_c = np.sum(tx_c) * dS_c
    L_c = np.sum(ty_c) * dS_c

    # --- Beam edges ---
    x_att = xc + r
    x_end = x_att + beam_L
    y_top = yc + 0.5 * beam_h
    y_bot = yc - 0.5 * beam_h
    ds_e  = (x_end - x_att) / n_edge
    ds_r  = beam_h / n_edge

    xs_top = np.linspace(x_att, x_end, n_edge, endpoint=False)
    tx_t, ty_t = _traction(model, xs_top, np.full_like(xs_top, y_top),
                            np.zeros(n_edge), np.ones(n_edge))
    xs_bot = np.linspace(x_att, x_end, n_edge, endpoint=False)
    tx_b, ty_b = _traction(model, xs_bot, np.full_like(xs_bot, y_bot),
                            np.zeros(n_edge), -np.ones(n_edge))
    ys_r = np.linspace(y_bot, y_top, n_edge, endpoint=False)
    tx_r, ty_r = _traction(model, np.full_like(ys_r, x_end), ys_r,
                            np.ones(n_edge), np.zeros(n_edge))

    D_beam = (np.sum(tx_t) + np.sum(tx_b)) * ds_e + np.sum(tx_r) * ds_r
    L_beam = (np.sum(ty_t) + np.sum(ty_b)) * ds_e + np.sum(ty_r) * ds_r

    D = D_c + D_beam
    L = L_c + L_beam

    ref = 0.5 * rho * Ubar ** 2 * d
    Cd  = D / ref if ref != 0 else float('nan')
    Cl  = L / ref if ref != 0 else float('nan')
    return D, L, Cd, Cl


# ============================================================
# Main entry-point
# ============================================================

def build_problem(Ubar=0.2, n_collo=4000, n_refine=1000,
                  n_wall=441, n_inlet=201, n_outlet=201,
                  n_cyl=400, n_beam=300):
    """Build all boundary and collocation point arrays.

    Parameters
    ----------
    Ubar : mean inlet velocity  (CFD1=0.2, CFD2=1.0, CFD3=2.0)
    """
    L, H = 1.1, 0.41
    xc, yc, r = 0.2, 0.2, 0.05

    # Beam bounding box (Turek-Hron)
    x_right, y_bot = 0.6, 0.19
    x_left          = x_right - 0.35   # 0.25
    y_top           = y_bot   + 0.02   # 0.21

    lb = np.array([0.0, 0.0])
    ub = np.array([L,   H])
    U_max = 1.5 * Ubar  # parabolic peak → mean = (2/3) U_max = Ubar

    # --- Boundary points ---
    wall_top = np.hstack([L * lhs(1, n_wall), H * np.ones((n_wall, 1))])
    wall_bot = np.hstack([L * lhs(1, n_wall), np.zeros((n_wall, 1))])

    y_in  = H * lhs(1, n_inlet)
    INLET = np.hstack([np.zeros((n_inlet, 1)), y_in,
                       4 * U_max * y_in * (H - y_in) / H ** 2,
                       np.zeros((n_inlet, 1))])

    OUTLET = np.hstack([L * np.ones((n_outlet, 1)), H * lhs(1, n_outlet)])

    theta_c = 2 * np.pi * lhs(1, n_cyl)
    CYLD = np.hstack([xc + r * np.cos(theta_c), yc + r * np.sin(theta_c)])

    beam_top_pts   = np.hstack([x_left + (x_right - x_left) * lhs(1, n_beam),
                                 y_top  * np.ones((n_beam, 1))])
    beam_bot_pts   = np.hstack([x_left + (x_right - x_left) * lhs(1, n_beam),
                                 y_bot  * np.ones((n_beam, 1))])
    beam_right_pts = np.hstack([x_right * np.ones((n_beam, 1)),
                                 y_bot  + (y_top - y_bot) * lhs(1, n_beam)])
    BEAM = np.vstack([beam_top_pts, beam_bot_pts, beam_right_pts])

    WALL = np.vstack([CYLD, BEAM, wall_top, wall_bot])

    # --- Collocation points (interior) ---
    XY_c        = lb + (ub - lb) * lhs(2, n_collo)
    XY_refine   = np.array([0.1, 0.1]) + np.array([0.6, 0.2]) * lhs(2, n_refine)
    XY_c        = np.vstack([XY_c, XY_refine])
    XY_c        = DelObsPT(XY_c, xc, yc, r, x_left, x_right, y_bot, y_top)
    XY_c        = np.vstack([XY_c, WALL, OUTLET, INLET[:, :2]])

    print(f"Collocation points: {XY_c.shape[0]}")
    return XY_c, INLET, OUTLET, WALL, lb, ub, (xc, yc, r, x_left, x_right, y_bot, y_top)


def main():
    # ---- Fluid properties (Turek-Hron CFD1: Re ≈ 20) ----
    rho   = 1.0
    nu    = 0.02        # kinematic viscosity
    mu    = rho * nu    # dynamic viscosity
    Ubar  = 0.2         # mean inlet velocity

    # ---- Problem setup ----
    uv_layers = [2] + 8 * [40] + [5]
    XY_c, INLET, OUTLET, WALL, lb, ub, geom = build_problem(Ubar=Ubar)

    # ---- Weight configuration (Strategy: momentum-heavy, constitutive-light) ----
    # See weighting strategy notes in the notebook for alternatives.
    weights = {
        'w_fu':    2.0,   # momentum harder to satisfy → boost
        'w_fv':    2.0,
        'w_fs11':  0.5,   # constitutive eqs are algebraic → relax
        'w_fs22':  0.5,
        'w_fs12':  0.5,
        'w_fp':    1.0,
        'w_wall':  10.0,
        'w_inlet': 10.0,
        'w_outlet': 5.0,
    }

    # ---- Training ----
    model = PINN_beam_flow_TF2(
        XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub,
        rho=rho, mu=mu, weights=weights
    )
    model.print_weights()

    print("=== Adam optimisation ===")
    t0 = time.time()
    model.train_adam(iters=10000, lr=5e-4, print_every=200)
    n_adam = len(model.loss_total)

    print("=== L-BFGS-B refinement ===")
    model.train_lbfgs(maxiter=50000)
    print(f"Total training time: {time.time() - t0:.1f} s")

    model.save('uvNN_beam_weights.weights.h5')

    # ---- Loss plot ----
    plot_loss_all(model, n_adam=n_adam)

    # ---- Field visualisation ----
    xc, yc, r, xb0, xb1, yb0, yb1 = geom
    L, H = ub
    nx, ny = 251, 101
    xg = np.linspace(0, L, nx); yg = np.linspace(0, H, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    xs = Xg.ravel()[:, None]; ys = Yg.ravel()[:, None]
    dst_cyl  = np.sqrt((xs - xc) ** 2 + (ys - yc) ** 2)
    in_beam  = ((xs[:, 0] >= xb0) & (xs[:, 0] <= xb1) &
                (ys[:, 0] >= yb0) & (ys[:, 0] <= yb1))
    mask = (dst_cyl[:, 0] >= r) & ~in_beam
    x_ev, y_ev = xs[mask], ys[mask]
    u_p, v_p, p_p = model.predict(x_ev, y_ev)
    postProcess(0, L, 0, H,
                field_pinn=[x_ev, y_ev, u_p, v_p, p_p])

    # ---- Drag / Lift ----
    D, L_f, Cd, Cl = compute_drag_lift(model, rho=rho, Ubar=Ubar)
    print(f"Drag D = {D:.6f}   Lift L = {L_f:.6f}")
    print(f"Cd = {Cd:.4f}   Cl = {Cl:.4f}")

    # ---- Save results ----
    results = {
        'loss_total':  model.loss_total,
        'loss_fu':     model.loss_fu,    'loss_fv':    model.loss_fv,
        'loss_fs11':   model.loss_fs11,  'loss_fs22':  model.loss_fs22,
        'loss_fs12':   model.loss_fs12,  'loss_fp':    model.loss_fp,
        'loss_pde':    model.loss_pde,
        'loss_wall':   model.loss_wall,
        'loss_inlet':  model.loss_inlet,
        'loss_outlet': model.loss_outlet,
        'u_pred': u_p, 'v_pred': v_p, 'p_pred': p_p,
        'x_eval': x_ev, 'y_eval': y_ev,
        'drag': D, 'lift': L_f, 'Cd': Cd, 'Cl': Cl,
        'meta': {
            'rho': rho, 'mu': mu, 'Ubar': Ubar,
            'weights': model._weights_dict,
            'layers': uv_layers,
            'description': 'Steady cylinder+beam PINN (Turek-Hron CFD1)'
        }
    }
    with open('pinn_beam_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Results saved to pinn_beam_results.pkl")


if __name__ == '__main__':
    main()
