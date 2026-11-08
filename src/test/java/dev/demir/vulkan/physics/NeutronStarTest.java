package dev.demir.vulkan.physics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Light leaving a neutron star surface (radius R ≥ 3M), checked against the exact orbit integral
 * and the apparent-size formula. Units: G = c = 1, M = 1. Numbers: docs/neutron-star.md.
 */
class NeutronStarTest {

    private static final double M = 1.0;

    /** Exact ψ(α) = ∫₀^{1/R} du / √(1/b² - u²(1 - 2Mu)), b = R sin α / √(1 - 2M/R). */
    private static double psiExact(double radius, double alpha) {
        double uR = 1.0 / radius;
        double b = radius * Math.sin(alpha) / Math.sqrt(1.0 - 2.0 * M / radius);
        int n = 200_000;
        double sum = 0;
        for (int i = 0; i < n; i++) { // u = uR (1 - s²) removes the singularity at α = 90°
            double s = (i + 0.5) / n;
            double u = uR * (1.0 - s * s);
            double f = 1.0 / (b * b) - u * u * (1.0 - 2.0 * M * u);
            sum += 2.0 * uR * s / Math.sqrt(f) / n;
        }
        return sum;
    }

    /** ψ measured with the integrator: emit at (R, 0, 0) at angle α from the normal, trace far away. */
    private static double psiIntegrated(double radius, double alpha) {
        double[] x0 = {radius, 0, 0};
        double[] n = {Math.cos(alpha), 0, Math.sin(alpha)};
        Schwarzschild.Result result = Schwarzschild.trace(x0, n, M, 1e6, 1_000_000);
        assertEquals(Schwarzschild.Fate.ESCAPED, result.fate());
        double[] d = result.direction();
        return Math.atan2(Math.sqrt(d[1] * d[1] + d[2] * d[2]), d[0]); // angle from +x (the emission normal)
    }

    @Test
    void farthestVisiblePointMatchesTheExactIntegral() {
        for (double radius : new double[]{5.8, 8.0, 12.0}) {
            for (double alphaDeg : new double[]{45.0, 90.0}) {
                double alpha = Math.toRadians(alphaDeg);
                double exact = psiExact(radius, alpha);
                double integrated = psiIntegrated(radius, alpha);
                System.out.printf("R = %4.1fM, alpha = %2.0f deg: psi integrator %.5f deg, exact %.5f deg%n",
                        radius, alphaDeg, Math.toDegrees(integrated), Math.toDegrees(exact));
                // Remaining bending beyond r = 1e6 M is ~1e-6 rad.
                assertEquals(exact, integrated, 2e-5, "R = " + radius + ", alpha = " + alphaDeg);
            }
        }
        // The headline number: 75.5 % of the surface of a 1.4 solar-mass, 12 km star is visible.
        double visible = (1 - Math.cos(psiIntegrated(5.8, Math.PI / 2))) / 2;
        assertEquals(0.755, visible, 0.001);
    }

    /** Angle from the direction to the star at which a camera at distance ro stops seeing the surface. */
    private static double apparentRadius(double mass, double radius, double ro) {
        double lo = 0, hi = 0.6;
        for (int i = 0; i < 50; i++) {
            double theta = 0.5 * (lo + hi);
            double[] x0 = {0, 0, -ro};
            double[] n = {Math.sin(theta), 0, Math.cos(theta)};
            boolean hits = Schwarzschild.trace(x0, n, mass, 100.0, 100_000, radius).fate()
                    == Schwarzschild.Fate.HIT_SURFACE;
            if (hits) lo = theta; else hi = theta;
        }
        return 0.5 * (lo + hi);
    }

    @Test
    void apparentSizeMatchesTheFormula() {
        double radius = 5.8, ro = 30.0;
        double formula = Math.asin(radius / ro * Math.sqrt(1 - 2 * M / ro) / Math.sqrt(1 - 2 * M / radius));
        double measured = apparentRadius(M, radius, ro);
        double straight = apparentRadius(0.0, radius, ro); // no gravity: plain geometry
        System.out.printf("apparent radius at 30M: integrator %.5f deg, formula %.5f deg; straight %.5f deg (asin(R/r) %.5f)%n",
                Math.toDegrees(measured), Math.toDegrees(formula), Math.toDegrees(straight), Math.toDegrees(Math.asin(radius / ro)));
        // One pixel at 40 deg vertical FOV, 720 px is ~9.7e-4 rad.
        assertEquals(formula, measured, 5e-5);
        assertEquals(Math.asin(radius / ro), straight, 5e-5);
    }
}
