//  Copyright (c) 2026 Scripps Research, Forli Lab.
//  All rights reserved.
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the Free Software
//  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mcswell::analysis {

// Brent's method for a bracketed root, following the classic
// bisection/secant/inverse-quadratic-interpolation hybrid (the same
// algorithm scipy.optimize.brentq implements). Defaults match scipy's:
// xtol=2e-12, rtol=4*machine epsilon, maxiter=100.
template <class F>
double brentq(
    F f, double xa, double xb, double xtol = 2e-12,
    double rtol = 8.881784197001252e-16, int maxiter = 100) {
    double fpre = f(xa);
    double fcur = f(xb);
    if (fpre == 0.0) return xa;
    if (fcur == 0.0) return xb;
    if ((fpre > 0.0) == (fcur > 0.0)) {
        throw std::runtime_error("brentq: root is not bracketed by [xa, xb]");
    }

    double xpre = xa, xcur = xb;
    double xblk = 0.0, fblk = 0.0, spre = 0.0, scur = 0.0;

    for (int i = 0; i < maxiter; ++i) {
        if (fpre != 0.0 && fcur != 0.0 && (fpre > 0.0) != (fcur > 0.0)) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (std::fabs(fblk) < std::fabs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        const double delta = (xtol + rtol * std::fabs(xcur)) / 2.0;
        const double sbis = (xblk - xcur) / 2.0;

        if (fcur == 0.0 || std::fabs(sbis) < delta) return xcur;

        if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
            double stry;
            if (xpre == xblk) {
                // Secant step.
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                // Inverse quadratic interpolation.
                const double dpre = (fpre - fcur) / (xpre - xcur);
                const double dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2.0 * std::fabs(stry) < std::min(std::fabs(spre), 3.0 * std::fabs(sbis) - delta)) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if (std::fabs(scur) > delta) {
            xcur += scur;
        } else {
            xcur += (sbis > 0.0) ? delta : -delta;
        }

        fcur = f(xcur);
    }

    throw std::runtime_error("brentq: maximum iterations exceeded");
}

} // namespace mcswell::analysis
