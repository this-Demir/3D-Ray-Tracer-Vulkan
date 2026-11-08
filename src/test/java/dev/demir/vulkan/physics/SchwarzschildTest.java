package dev.demir.vulkan.physics;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks the black hole ray integrator (the CPU twin of the shader) against exact or well-known
 * results of Schwarzschild geometry. Units: G = c = 1, mass M = 1 (horizon at r = 2).
 */
class SchwarzschildTest {

    private static final double M = 1.0;
    private static final double B_CRITICAL = 3.0 * Math.sqrt(3.0) * M; // 5.196152...

    /** A ray starting at distance R on the -z axis whose impact parameter is exactly b. */
    private static double[][] rayWithImpactParameter(double b, double radius) {
        double sinTheta = b * Math.sqrt(1.0 - 2.0 * M / radius) / radius; // b = R sinθ / √(1 - 2M/R)
        double[] x0 = {0, 0, -radius};
        double[] n = {sinTheta, 0, Math.sqrt(1.0 - sinTheta * sinTheta)};
        return new double[][]{x0, n};
    }

    /**
     * Integrates from R inward and back out to R and returns the polar angle swept between the
     * start and the point where the ray crosses r = R again. That point is found by bisecting the
     * length of the last RK4 step (a straight-line interpolation between steps is not accurate
     * enough for this comparison).
     */
    private static double sweptAngle(double b, double radius) {
        double[][] ray = rayWithImpactParameter(b, radius);
        double[] x0 = ray[0];
        double[] v0 = Schwarzschild.initialVelocity(x0, ray[1], M);
        double[] s = {x0[0], x0[1], x0[2], v0[0], v0[1], v0[2]};
        double h2 = Schwarzschild.cross2(x0, v0);
        double phi = 0;
        double[] prev = {s[0], s[1], s[2]};
        boolean turned = false;
        for (int i = 0; i < 1_000_000; i++) {
            double r = Math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
            double[] before = s.clone();
            double dl = Schwarzschild.stepSize(r);
            Schwarzschild.step(s, dl, h2, M);
            double[] cur = {s[0], s[1], s[2]};
            if (s[0] * s[3] + s[1] * s[4] + s[2] * s[5] > 0) turned = true;
            if (turned && Schwarzschild.length(cur) >= radius) {
                double lo = 0, hi = dl;
                for (int it = 0; it < 60; it++) {
                    double mid = 0.5 * (lo + hi);
                    double[] t = before.clone();
                    Schwarzschild.step(t, mid, h2, M);
                    if (Schwarzschild.length(new double[]{t[0], t[1], t[2]}) < radius) lo = mid; else hi = mid;
                }
                double[] t = before.clone();
                Schwarzschild.step(t, 0.5 * (lo + hi), h2, M);
                return phi + angleBetween(prev, new double[]{t[0], t[1], t[2]});
            }
            phi += angleBetween(prev, cur);
            prev = cur;
        }
        throw new AssertionError("ray did not come back out");
    }

    /** Exact swept angle 2 ∫_{u_R}^{u_0} du / √(1/b² - u²(1 - 2Mu)) by quadrature. */
    private static double exactSweptAngle(double b, double radius) {
        double uR = 1.0 / radius;
        double u0 = turningPoint(b);
        // u = u0 - (u0 - uR) s²  removes the 1/√ singularity at the turning point s = 0.
        int n = 200_000;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            double sMid = (i + 0.5) / n;
            double u = u0 - (u0 - uR) * sMid * sMid;
            double f = 1.0 / (b * b) - u * u * (1.0 - 2.0 * M * u);
            sum += 2.0 * (u0 - uR) * sMid / Math.sqrt(f) / n;
        }
        return 2.0 * sum;
    }

    /** Smallest positive root of 1/b² - u²(1 - 2Mu) = 0 (the closest approach, u = 1/r_min). */
    private static double turningPoint(double b) {
        double lo = 0, hi = 1.0 / (3.0 * M);
        for (int i = 0; i < 200; i++) {
            double mid = 0.5 * (lo + hi);
            double f = 1.0 / (b * b) - mid * mid * (1.0 - 2.0 * M * mid);
            if (f > 0) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    private static double angleBetween(double[] a, double[] b) {
        double cross = Math.sqrt(Schwarzschild.cross2(a, b));
        return Math.atan2(cross, Schwarzschild.dot(a, b));
    }

    @Test
    void initialVelocityGivesTheRequestedImpactParameter() {
        for (double b : new double[]{2.0, B_CRITICAL, 10.0, 100.0}) {
            double[][] ray = rayWithImpactParameter(b, 1000.0); // b must be < R / √(1 - 2M/R)
            double[] v0 = Schwarzschild.initialVelocity(ray[0], ray[1], M);
            assertEquals(b, Schwarzschild.impactParameter(ray[0], v0, M), 1e-12 * b, "b = " + b);
        }
    }

    @Test
    void conservesAngularMomentumAndEnergy() {
        double[][] ray = rayWithImpactParameter(6.0, 40.0);
        double[] x0 = ray[0];
        double[] v0 = Schwarzschild.initialVelocity(x0, ray[1], M);
        double[] s = {x0[0], x0[1], x0[2], v0[0], v0[1], v0[2]};
        double h2 = Schwarzschild.cross2(x0, v0);
        double c0 = Schwarzschild.energy(x0, v0, h2, M);
        double maxDriftH = 0, maxDriftC = 0;
        for (int i = 0; i < 500; i++) {
            double[] x = {s[0], s[1], s[2]};
            Schwarzschild.step(s, Schwarzschild.stepSize(Schwarzschild.length(x)), h2, M);
            double[] xn = {s[0], s[1], s[2]};
            double[] vn = {s[3], s[4], s[5]};
            maxDriftH = Math.max(maxDriftH, Math.abs(Schwarzschild.cross2(xn, vn) / h2 - 1));
            maxDriftC = Math.max(maxDriftC, Math.abs(Schwarzschild.energy(xn, vn, h2, M) / c0 - 1));
        }
        System.out.printf("conservation drift over 500 steps: h %.2e, C %.2e%n", maxDriftH, maxDriftC);
        assertTrue(maxDriftH < 1e-6, "h drift " + maxDriftH);
        assertTrue(maxDriftC < 1e-6, "C drift " + maxDriftC);
    }

    @Test
    void pathMatchesTheExactOrbitIntegral() {
        for (double b : new double[]{5.3, 6.0, 10.0, 30.0, 100.0}) {
            double numeric = sweptAngle(b, 40.0 + b);
            double exact = exactSweptAngle(b, 40.0 + b);
            System.out.printf("b = %6.2f  swept angle: integrator %.9f, exact %.9f, diff %.1e%n",
                    b, numeric, exact, numeric - exact);
            // One pixel of the 1280x720 view at 30 deg vertical FOV is ~7.3e-4 rad; 1e-7 rad is
            // far below that (measured agreement is ~1e-9).
            assertEquals(exact, numeric, 1e-7, "b = " + b);
        }
    }

    @Test
    void weakFieldDeflectionMatchesTheSeries() {
        for (double b : new double[]{100.0, 1000.0}) {
            double radius = 1e6;
            // Swept angle = π + α - (bending missing beyond R) - 2 asin-like flat-space part.
            double alpha = sweptAngle(b, radius) - Math.PI + 2.0 * Math.asin(b / radius);
            double series = 4.0 * M / b + 15.0 * Math.PI / 4.0 * (M / b) * (M / b);
            System.out.printf("b = %6.0f  deflection %.7f, 2nd-order series %.7f%n", b, alpha, series);
            // Neglected: the 3rd-order term (~43 (M/b)^3) and the bending beyond R (~4M/R).
            assertEquals(series, alpha, 60.0 * Math.pow(M / b, 3) + 8.0 * M / radius, "b = " + b);
        }
    }

    @Test
    void captureThresholdIsTheCriticalImpactParameter() {
        double[][] inside = rayWithImpactParameter(B_CRITICAL * (1 - 1e-4), 40.0);
        double[][] outside = rayWithImpactParameter(B_CRITICAL * (1 + 1e-4), 40.0);
        assertEquals(Schwarzschild.Fate.CAPTURED,
                Schwarzschild.trace(inside[0], inside[1], M, 100.0, 100_000).fate());
        assertEquals(Schwarzschild.Fate.ESCAPED,
                Schwarzschild.trace(outside[0], outside[1], M, 100.0, 100_000).fate());
    }

    @Test
    void criticalRayGrazesThePhotonSphere() {
        double[][] ray = rayWithImpactParameter(B_CRITICAL * (1 + 1e-9), 40.0);
        Schwarzschild.Result result = Schwarzschild.trace(ray[0], ray[1], M, 100.0, 100_000);
        System.out.printf("critical ray: %s, closest approach r = %.6f (photon sphere 3M)%n",
                result.fate(), result.minRadius());
        assertEquals(Schwarzschild.Fate.ESCAPED, result.fate());
        assertEquals(3.0 * M, result.minRadius(), 1e-3);
    }

    @Test
    void shadowSizeMatchesSyngesFormula() {
        double observer = 40.0;
        // Bisect the angle (from the direction towards the hole) where rays stop being captured.
        double lo = 0, hi = 0.5;
        for (int i = 0; i < 60; i++) {
            double alpha = 0.5 * (lo + hi);
            double[] x0 = {0, 0, -observer};
            double[] n = {Math.sin(alpha), 0, Math.cos(alpha)};
            boolean captured = Schwarzschild.trace(x0, n, M, 100.0, 100_000).fate() == Schwarzschild.Fate.CAPTURED;
            if (captured) lo = alpha; else hi = alpha;
        }
        double numeric = 0.5 * (lo + hi);
        double synge = Math.asin(B_CRITICAL * Math.sqrt(1.0 - 2.0 * M / observer) / observer);
        System.out.printf("shadow angular radius at r = 40M: integrator %.7f rad, Synge %.7f rad (%.4f deg)%n",
                numeric, synge, Math.toDegrees(synge));
        assertEquals(synge, numeric, 1e-6);
    }
}
