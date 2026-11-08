package dev.demir.vulkan.physics;

/**
 * Light paths around a non-rotating (Schwarzschild) black hole, in units G = c = 1.
 *
 * This is the CPU reference of the ray marching in pathtrace.comp (same equations, same step
 * rule), used by the unit tests to check the physics against known results.
 *
 * Coordinates are Cartesian, centered on the black hole. A light ray is integrated as
 *
 *     dx/dλ = v,     dv/dλ = -3 M h² x / r⁵,     h = |x × v|  (constant)
 *
 * which traces exactly the spatial path of a Schwarzschild null geodesic in Schwarzschild
 * coordinates (Binet's equation turns this force law into u'' + u = 3 M u², the photon orbit
 * equation; see docs/black-hole.md for the derivation).
 */
public final class Schwarzschild {

    /** Step length as a fraction of the distance to the hole (dλ = STEP_FACTOR · r). */
    public static final double STEP_FACTOR = 0.02;

    private Schwarzschild() {
    }

    /** Outcome of tracing one ray. */
    public enum Fate { CAPTURED, ESCAPED, HIT_SURFACE, MAX_STEPS }

    /**
     * @param fate      what happened to the ray
     * @param direction unit direction after escaping (or the last direction)
     * @param minRadius closest approach to the hole
     * @param steps     integration steps used
     */
    public record Result(Fate fate, double[] direction, double minRadius, int steps) {
    }

    /**
     * Converts the direction a static observer at x sees a ray leave in (unit vector n, measured
     * in the observer's local frame) into the integration velocity: the radial component is
     * scaled by √(1 - 2M/r). With this, h = r n_⊥ = L and ½|v|² - M h²/r³ = ½ E² with
     * E² = 1 - 2M/r, so the impact parameter is b = L/E = r n_⊥ / √(1 - 2M/r).
     */
    public static double[] initialVelocity(double[] x, double[] n, double mass) {
        double r = length(x);
        double radial = dot(n, x) / r;
        double shrink = 1.0 - Math.sqrt(1.0 - 2.0 * mass / r);
        return new double[]{
                n[0] - shrink * radial * x[0] / r,
                n[1] - shrink * radial * x[1] / r,
                n[2] - shrink * radial * x[2] / r};
    }

    /** Impact parameter b = h / E of a state, with E² = |v|² - 2 M h² / r³. */
    public static double impactParameter(double[] x, double[] v, double mass) {
        double h2 = cross2(x, v);
        double r = length(x);
        double e2 = dot(v, v) - 2.0 * mass * h2 / (r * r * r);
        return Math.sqrt(h2 / e2);
    }

    /** The conserved "energy" C = ½|v|² - M h²/r³ (equals ½E²). */
    public static double energy(double[] x, double[] v, double h2, double mass) {
        double r = length(x);
        return 0.5 * dot(v, v) - mass * h2 / (r * r * r);
    }

    /** dλ for a ray at distance r: proportional to r, so the far field takes few long steps. */
    public static double stepSize(double r) {
        return STEP_FACTOR * r;
    }

    /**
     * One classic Runge–Kutta (RK4) step of length dl. state = {x, y, z, vx, vy, vz}, h2 = h².
     */
    public static void step(double[] state, double dl, double h2, double mass) {
        double[] k1 = derivative(state, h2, mass);
        double[] k2 = derivative(add(state, k1, dl / 2), h2, mass);
        double[] k3 = derivative(add(state, k2, dl / 2), h2, mass);
        double[] k4 = derivative(add(state, k3, dl), h2, mass);
        for (int i = 0; i < 6; i++) {
            state[i] += dl / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        }
    }

    /**
     * Traces a ray from x0 (outside the photon sphere) leaving in local direction n until it is
     * captured or moves outward beyond escapeRadius.
     *
     * Capture rule: a ray inside the photon sphere (r < 3M) that moves inward never turns back.
     * Proof: (du/dφ)² = 1/b² - V(u) with V(u) = u²(1 - 2Mu), and V decreases for u > 1/(3M).
     * Moving inward, u grows and 1/b² ≥ V(u) already holds, so 1/b² > V(u') for every larger u'
     * and there is no turning point before the horizon. Rays that escape always turn around
     * outside 3M, so a step proportional to r stays well-behaved for every ray that matters.
     */
    public static Result trace(double[] x0, double[] n, double mass, double escapeRadius, int maxSteps) {
        return trace(x0, n, mass, escapeRadius, maxSteps, 0.0);
    }

    /**
     * Like {@link #trace(double[], double[], double, double, int)}, for a compact object with a
     * surface of radius surfaceRadius ≥ 3M (a neutron star; 0 = black hole). Each step's straight
     * chord is intersected exactly with the surface sphere, as the shader does.
     */
    public static Result trace(double[] x0, double[] n, double mass, double escapeRadius, int maxSteps,
                               double surfaceRadius) {
        double[] v0 = initialVelocity(x0, n, mass);
        double[] state = {x0[0], x0[1], x0[2], v0[0], v0[1], v0[2]};
        double h2 = cross2(x0, v0);
        double minRadius = length(x0);
        for (int i = 0; i < maxSteps; i++) {
            double r = Math.sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
            minRadius = Math.min(minRadius, r);
            double outward = state[0] * state[3] + state[1] * state[4] + state[2] * state[5];
            if (surfaceRadius == 0 && r < 3.0 * mass && outward < 0) {
                return new Result(Fate.CAPTURED, direction(state), minRadius, i);
            }
            if (r > escapeRadius && outward > 0) {
                return new Result(Fate.ESCAPED, direction(state), minRadius, i);
            }
            double[] before = {state[0], state[1], state[2]};
            step(state, stepSize(r), h2, mass);
            if (surfaceRadius > 0 && chordHitsSphere(before, new double[]{state[0], state[1], state[2]}, surfaceRadius)) {
                return new Result(Fate.HIT_SURFACE, direction(state), surfaceRadius, i + 1);
            }
        }
        return new Result(Fate.MAX_STEPS, direction(state), minRadius, maxSteps);
    }

    /** True if the straight segment a -> b passes inside the sphere of the given radius around the origin. */
    static boolean chordHitsSphere(double[] a, double[] b, double radius) {
        double[] d = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        double dd = dot(d, d);
        // closest point of the segment to the origin
        double t = Math.max(0, Math.min(1, -dot(a, d) / dd));
        double[] p = {a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]};
        return dot(p, p) <= radius * radius;
    }

    // --- small vector helpers ---

    private static double[] derivative(double[] s, double h2, double mass) {
        double r2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        double r5 = r2 * r2 * Math.sqrt(r2);
        double k = -3.0 * mass * h2 / r5;
        return new double[]{s[3], s[4], s[5], k * s[0], k * s[1], k * s[2]};
    }

    private static double[] add(double[] s, double[] d, double scale) {
        double[] out = new double[6];
        for (int i = 0; i < 6; i++) out[i] = s[i] + scale * d[i];
        return out;
    }

    private static double[] direction(double[] s) {
        double len = Math.sqrt(s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
        return new double[]{s[3] / len, s[4] / len, s[5] / len};
    }

    static double dot(double[] a, double[] b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    static double length(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    /** |a × b|². */
    static double cross2(double[] a, double[] b) {
        double cx = a[1] * b[2] - a[2] * b[1];
        double cy = a[2] * b[0] - a[0] * b[2];
        double cz = a[0] * b[1] - a[1] * b[0];
        return cx * cx + cy * cy + cz * cz;
    }
}
