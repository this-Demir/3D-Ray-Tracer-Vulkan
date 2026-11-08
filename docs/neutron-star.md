# Neutron Star: Physics, Visualization and Verification

A neutron star is a compact object without an event horizon: light leaves its surface and is bent
by its gravity on the way to the camera. The **Neutron Star** scene uses the same Schwarzschild light
paths as the black hole (see [black-hole.md](black-hole.md)); the difference is that rays can end on a
glowing surface. Units: G = c = 1, mass M.

## 1. Physics

A typical neutron star has 1.4 solar masses and a radius of about 12 km. In units of M = GM/c² ≈ 2.07 km
this is **R ≈ 5.8M**: outside the photon sphere (3M), so every ray that reaches the camera leaves the
surface directly.

**Bending.** A photon leaving the surface at angle α from the outward normal has impact parameter
b = R sin α / √(1 - 2M/R) (static emitter, see black-hole.md §2). The angle ψ between the emission point's
normal and the direction to a distant observer follows from the orbit equation (2) of black-hole.md:

    ψ(α) = ∫₀^{1/R} du / √(1/b² - u²(1 - 2Mu))

Beloborodov (2002) gives the simple approximation **1 - cos α = (1 - cos ψ)(1 - 2M/R)**.

**How much of the surface is visible.** The farthest visible point is reached by a photon leaving
tangentially (α = 90°). Without gravity ψ_max = 90° and exactly half the surface is visible. Computed
from the exact integral:

| R / M | ψ_max (exact) | ψ_max (Beloborodov) | visible surface |
|---|---|---|---|
| 12 | 101.45° | 101.54° | 59.9 % |
| 8 | 109.17° | 109.47° | 66.4 % |
| **5.8** | **120.68°** | 121.76° | **75.5 %** |
| 4 | 152.57° | 180° | 94.4 % |
| 3.5 | 181.8° | – | 100 % (some points are seen twice) |

**Apparent size.** The star looks larger than it is: the outermost ray grazes the surface, so seen from
a static observer at distance r_o its angular radius is

    sin θ = (R / r_o) √(1 - 2M/r_o) / √(1 - 2M/R)

For R = 5.8M, r_o = 30M: **θ = 13.34°** instead of 11.15° for straight light; with a 40° vertical field
of view at 1280×720 that is **234.6 px** instead of **194.9 px**.

**Gravitational redshift.** Light climbing out of the potential is shifted by g = √(1 - 2M/R) = 0.809:
temperatures look 19 % lower and the brightness (specific intensity, which scales with g⁴) drops to 43 %.

**Pulses.** A hot spot at colatitude θ_s on a star whose spin axis is inclined by i to the line of sight
has cos ψ = cos i cos θ_s + sin i sin θ_s cos φ over the spin phase φ; its flux is proportional to
cos α · d(cos α)/d(cos ψ). For θ_s = 45°, i = 60° (relative to the maximum):

| phase | 0° | 60° | 120° | 150° | 180° |
|---|---|---|---|---|---|
| straight light | 1.000 | 0.683 | 0.049 | **0** | **0** |
| exact (R = 5.8M) | 1.000 | 0.795 | 0.386 | 0.235 | **0.179** |
| Beloborodov | 1.000 | 0.795 | 0.384 | 0.234 | 0.179 |

Without bending the spot would vanish for about 40 % of each turn; with bending it never disappears.
(Pechenick, Ftaclas & Cohen 1983 for the exact treatment; Beloborodov 2002 for the approximation.)

## 2. Visualization

The scene has to make an invisible effect obvious, so some elements are physical and some are
deliberately diagnostic:

| Element | Kind | Why |
|---|---|---|
| Light bending, apparent size, lensed star field | physical | exact ray marching |
| Gravitational redshift of the surface (g⁴ brightness, g · temperature) | physical | uniform dimming / reddening |
| Hot spots at the magnetic poles, black-body colour | physical model, simplified | the classic pulsar picture |
| **Near / far side colouring** of the surface (cool blue facing the camera, warm orange facing away) with a latitude–longitude grid | diagnostic | every orange pixel is a part of the star that straight light could never show |
| **Split view**: left half straight light, right half bent light, same frame | diagnostic | direct comparison without toggling |
| Pulsar beams: glowing cones along the magnetic axis | artistic | shows where the radiation goes; the glow is sampled along the bent ray, so the beams look bent near the star |

The star spins with *Play animation*, so the hot spots move to the far side and stay visible.

## 3. Implementation plan

- One general ray march for compact objects (replaces the black hole's separate straight-ray path):
  integrate with gravity on, or march the same steps without acceleration when gravity is off.
- Per step, besides the scene geometry and the disk, intersect the chord with the star's surface sphere
  (radius R; for a black hole without gravity, the horizon 2M). The photon-sphere capture rule is only
  used for a black hole with gravity.
- Frame UBO grows by one vec4: surface radius, spin period, magnetic tilt, display flags (split view, beams).
- Tests: the integrator's ψ_max and apparent size against the exact values above.
- GPU check: measure the star's radius in the render (on and off), and count far-side pixels
  (none expected without gravity).

## 4. Verification

*(Filled in after implementation.)*
