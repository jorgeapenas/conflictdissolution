module ConflictDissolution

using SymPy
using LinearAlgebra

♀ = 1
♂ = 2
mr = 1
rm = 2

struct ModelParameters

    P::String # ploidy/genetic system
    G::String # helpers' sex
    σ1 # proportion of females 1st brood
    σ2 # proportion of females 2nd brood
    f1::Int64 # early fertility (number of offspring 1st brood)
    s1 # 1st brood survival
    s2 # 2nd brood survival
    f0 # parameter for f2(z) function
    α # parameter for f2(z) function
    sMmin # maximum sM with 0 helpers
    sMmax # maximum sM
    f2min::Int64 # minimum maximum late fertility
    f2max::Int64 # maximum late fertility
    ξ # mother's power parameter
    ψ # offspring's power parameter
    contest_type::String # type of contest
    Gx # genetic variance x
    Gy # genetic variance y
    Gz # genetic variance z

end

struct ModelFunctions

    sM # mother survival probability function, sM(f2,h)
    f2 # late fertility function, f2(z)
    h # expected number of helpers function, h(p)
    Π2 # late productivity function, Π2(f2,h)
    zstar # optimal reproductive effort function, z*(h)
    f2star # optimal late fertility, f2*(h)
    B # helping benefit, B(f2,h)
    C # helping cost, (constant)
    D # marginal productivity of fertility, D(f2,h)
    u # stable distribution, u(f2,h)
    v # reproductive values, v(f2,h)
    p # joint phenotype function, p(x,y)
    dpdx # derivative of p with respect to x, dpdx(x,y)
    dpdy # derivative of p with respect to y, dpdx(x,y)
    BCO # critical benefit-cost ratio offspring perspective, (B/C)*(f2,h)
    BCM # critical benefit-cost ratio mother perspective, (B/C)*(f2,h)
    SpO # selection gradient for helping p under offspring control, SpO(p,z)
    SpM # selection gradient for helping p under maternal control, SpM(p,z)
    Sx # selection gradient for maternal influence x, Sx(x,y,z)
    Sy # selection gradient for offspring resistance y, Sy(x,y,z)
    Sz # selection gradient for reproductive effort z, Sz(x,y,z)
    dxdt # evolutionary dynamic for maternal influence x, dxdt(x,y,z)
    dydt # evolutionary dynamic for offspring resistance y, dxdt(x,y,z)
    dzdt # evolutionary dynamic for reproductive effort z, dxdt(x,y,z)
    evoldyn # evolutionary dynamics of x, y, and z
    evoldyn_z0 # evolutionary dynamics of x, y when z does not evolve (i.e., z=z*(0))
    evoldyn_zfast # evolutionary dynamics of x, y with fast z (i.e., z = z*(h))

end

Base.show(io::IO, mf::ModelFunctions) = print(io, "Model Functions")

# transmission probability matrix (eq. S.2.6.26 and Fig. S4)
function transmission_prob_matrix(m::ModelParameters)
    q = zeros((2,2))
    q[♀,mr] = 1/2
    q[♀,rm] = (1/2)*(m.P == "D") + 1*(m.P == "HD")
    q[♂,mr] = 1/2
    q[♂,rm] = (1/2)*(m.P == "D") + 0*(m.P == "HD")
    return q
end

# brood sex proportion vectors (eq. S.1.1.4)
function broodsexproportions(m::ModelParameters)
    σ1 = (m.σ1,1-m.σ1)
    σ2 = (m.σ2,1-m.σ2)
    return σ1, σ2
end

# maximum number of helpers (eq. S.1.1.5)
function hbar(m::ModelParameters)
    if m.G == "B"
        m.f1
    elseif m.G == "F"
        m.f1*m.σ1
    end
end

function setupfunctions(m::ModelParameters)

    # store maximum number of helpers
    h̄ = hbar(m)

    # expected number of helpers function, h(p)
    h_fn(p) = h̄*p

    # store brood sex proportion vectors and transmission prob matrix
    (σ1, σ2) = broodsexproportions(m)
    q = transmission_prob_matrix(m)

    ## set function for the vital rates

    # declare symbolic variables: number of helpers (h), late fertility effort (f2)
    h_symb, f2_symb = symbols("h f_2")

    # declare symbolic parameter: maximum number of helpers
    hbar_symb = symbols("\\bar{h}")

    # declare symbolic parameters for sMbar
    sMmin_symb, sMmax_symb = symbols("\\underline{s_M} \\overline{\\overline{s_M}}")

    # declare symbolic parameters for f2bar
    f2min_symb, f2max_symb = symbols("\\underline{f_2} \\overline{\\overline{f_2}}")

    # define sMbar and f2bar (eq. S.7.1.2a and S.7.1.2b)
    sMbar_symb = sMmin_symb + (sMmax_symb - sMmin_symb)*(h_symb/hbar_symb)
    f2bar_symb = f2min_symb + (f2max_symb - f2min_symb)*(h_symb/hbar_symb)

    # define mated pair survival function (eq. S7.1.1b)
    sM_symb = sMbar_symb - (sMbar_symb/f2bar_symb)*f2_symb

    # define second-brood survival constant (eq. S7.1.1c)
    s2_symb = symbols("s_2")

    # evaluate parameters
    sMbar_symb_eval = sMbar_symb.subs([(hbar_symb,h̄),(sMmin_symb,m.sMmin),(sMmax_symb,m.sMmax)])
    f2bar_symb_eval = f2bar_symb.subs([(hbar_symb,h̄),(f2min_symb,m.f2min),(f2max_symb,m.f2max)])
    sM_symb_eval = sM_symb.subs([(hbar_symb,h̄),(sMmin_symb,m.sMmin),(sMmax_symb,m.sMmax),(f2min_symb,m.f2min),(f2max_symb,m.f2max)])
    s2_symb_eval = s2_symb.subs(s2_symb, m.s2)

    # define and evaluate late productivity (eq. S7.1.2)
    Π2_symb = sM_symb*f2_symb*s2_symb
    Π2_symb_eval = sM_symb_eval*f2_symb*s2_symb_eval

    # lambdify expressions
    sM = lambdify(sM_symb_eval)
    Π2 = lambdify(Π2_symb_eval)
    f2bar = lambdify(f2bar_symb_eval)

    # define f2(z) function (eq. S7.1.1a)
    f2_fn(z) = m.f0 * z^m.α
    # calculate its derivative and inverse
    df2dz(z) = m.f0*m.α*z^(1-m.α)
    f2inv(f2) = (f2/m.f0)^(1/m.α)

    # define optimal late fertility (eq. S7.1.5)
    f2star(h) = f2bar(h)/2

    # optimal reproductive effort (eq. S7.1.6)
    zstar(h) = f2inv(f2star(h))

    # store solitary-life optimal reproductive effort
    z0 = zstar(0)

    ## calculate B, C, D

    # calculate marginal productivity of helpers (marginal benefit of helping) by taking the derivative
    # (eq. S7.1.3)
    B_symb_eval = diff(Π2_symb_eval,h_symb)

    # calculate marginal productivity of late fertility by taking the derivative
    # (eq. S7.1.3)
    D_symb_eval = diff(Π2_symb_eval,f2_symb)

    # lambdify expressions
    B = lambdify(B_symb_eval) # marginal benefit of helping
    D = lambdify(D_symb_eval) # marginal productivity of reproductive effort

    # define the marginal cost of helping (eq. S2.8.4)
    C = m.s1 # marginal cost of helping

    ## set stable distribution and reproductive values

    # expected lifetime number of reproductives (eq. S1.6.18)
    prrr(l,h) = (m.G == "B")*(h/h̄) + (m.G == "F")*(l == ♀)*(h/h̄)
    Π(l,f2,h) = σ1[l]*m.f1*(1-prrr(l,h))*m.s1 + σ2[l]*Π2(f2,h)

    # stable sex distribution (eq. S2.6.22 and S2.6.23)
    function utilde(l)
        if l == ♀
            q[♀,rm]/(q[♀,rm]+q[♂,mr])
        elseif l == ♂
            q[♂,mr]/(q[♀,rm]+q[♂,mr])
        end
    end

    # stable distribution (eq. S2.6.14 and S2.6.20)
    function u(f2,h)
        u1rm = utilde(♂)
        u1mr = utilde(♀)
        u2rm = u1rm*sM(f2,h)
        u2mr = u1mr*sM(f2,h)
        u♀m = u1mr*Π(♀,f2,h)
        u♂m = u1rm*Π(♂,f2,h)
        (u♀m, u♂m, u1rm, u1mr, u2rm, u2mr)
    end

    # resident late effective fertility (eq. S1.5.4)
    F2(l,f2) = f2*σ2[l]*m.s2

    # reproductive values for unmated individuals (eq. 2.6.12)
    vtilde(l,f2,h) = (l == ♂)*1 + (l == ♀)*Π(♂,f2,h)/Π(♀,f2,h)

    # reproductive values
    function v(f2,h)
        v♂m = vtilde(♂,f2,h) # eq. 2.6.12
        v♀m = vtilde(♀,f2,h) # eq. 2.6.12
        v1rm = Π(♂,f2,h) # eq. S2.6.9
        v1mr = v1rm # eq. S2.6.9
        v2rm = q[♀,rm] * F2(♀,f2) * v♀m + q[♂,rm] * F2(♂,f2) # eq. 2.6.10
        v2mr = q[♀,mr] * F2(♀,f2) * v♀m + q[♂,mr] * F2(♂,f2) # eq. 2.6.10
        (v♀m, v♂m, v1rm, v1mr, v2rm, v2mr)
    end

    ## set joint phenotype functions and their derivatives

    # declare symbolic variables and parameters
    x_symb, y_symb  = symbols("x y")
    xi_symb, psi_symb = symbols("xi psi")

    # define impact functions (eq. S7.2.3)
    gM_symb = exp(xi_symb*x_symb)-1
    gO_symb = exp(psi_symb*y_symb)-1

    # define joint phenotype function
    if m.contest_type == "simultaneous"
        pxy_symb = gM_symb/(1+gM_symb+gO_symb) # eq. S7.2.1
    elseif m.contest_type == "sequential"
        pxy_symb = (gM_symb/(1+gM_symb))*(1-gO_symb/(1+gO_symb)) # eq. S7.2.2
    end

    # take derivatives
    dpdx_symb = simplify(diff(pxy_symb,x_symb))
    dpdy_symb = simplify(diff(pxy_symb,y_symb))

    # evaluate parameters
    pxy_symb_eval = pxy_symb.subs([(xi_symb,m.ξ), (psi_symb,m.ψ)])
    dpdx_symb_eval = dpdx_symb.subs([(xi_symb,m.ξ), (psi_symb,m.ψ)])
    dpdy_symb_eval = dpdy_symb.subs([(xi_symb,m.ξ), (psi_symb,m.ψ)])

    # lambdify expressions
    p_fn = lambdify(pxy_symb_eval)
    dpdx = lambdify(dpdx_symb_eval)
    dpdy = lambdify(dpdy_symb_eval)

    ## set structure coefficients

    # Offspring control, both sexes help (eq. S2.8.13)
    function ιpOB(f2,h) # eq. S2.8.13a
        ι = 0
        for l in (♀,♂)
            ι += σ1[l]*utilde(l)*vtilde(l,f2,h)
        end
        return ι
    end
    function κpOB(f2,h) # eq. S2.8.13b
        κ = 0
        for l in (♀,♂)
            for lp in (♀,♂)
                for k in (rm,mr)
                    κ += σ1[l]*σ2[lp]*utilde(k)*q[l,k]*q[lp,k]*vtilde(lp,f2,h)
                end
            end
        end
        return κ
    end

    # Offspring control, only females help (eq. S2.8.16)
    ιpOF(f2,h) = σ1[♀]*utilde(♀)*vtilde(♀,f2,h) # eq. S2.8.16a
    function κpOF(f2,h) # eq. S2.8.16b
        κ = 0
        for lp in (♀,♂)
            for k in (rm,mr)
                κ += σ2[lp]*utilde(k)*q[♀,k]*q[lp,k]*vtilde(lp,f2,h)
            end
        end
        κ *= σ1[♀]
        return κ
    end

    # Maternal control, both sexes help (eq. S2.8.19)
    function ιpMB(f2,h) # eq. S2.8.19a
        ι = 0
        for l in (♀,♂)
            ι += σ1[l]*q[l,mr]*vtilde(l,f2,h)
        end
        ι *= utilde(mr)
        return ι
    end
    function κpMB(f2,h) # eq. S2.8.19b
        κ = 0
        for lp in (♀,♂)
            κ += σ2[lp]*q[lp,mr]*vtilde(lp,f2,h)
        end
        κ *= utilde(mr)
        return κ
    end

    # Maternal control, only females help (eq. S2.8.22)
    ιpMF(f2,h) = σ1[♀]*utilde(mr)*q[♀,mr]*vtilde(♀,f2,h) # eq. S2.8.22a
    function κpMF(f2,h) # eq. S2.8.22b
        κ = 0
        for lp in (♀,♂)
            κ += σ2[lp]*q[lp,mr]*vtilde(lp,f2,h)
        end
        κ *= σ1[♀]*utilde(mr)
        return κ
    end

    function ιpO(f2,h)
        if m.G == "B"
            ιpOB(f2,h)
        elseif m.G == "F"
            ιpOF(f2,h)
        end
    end
    function ιpM(f2,h)
        if m.G == "B"
            ιpMB(f2,h)
        elseif m.G == "F"
            ιpMF(f2,h)
        end
    end
    function κpO(f2,h)
        if m.G == "B"
            κpOB(f2,h)
        elseif m.G == "F"
            κpOF(f2,h)
        end
    end
    function κpM(f2,h)
        if m.G == "B"
            κpMB(f2,h)
        elseif m.G == "F"
            κpMF(f2,h)
        end
    end

    function κz(f2,h) # eq. 2.9.5
        κ = 0
        for lp in (♀,♂)
            κ += σ2[lp]*q[lp,mr]*vtilde(lp,f2,h)
        end
        κ *= utilde(mr)
        return κ
    end

    # critical benefit-cost ratios (eq. S2.8.10)
    BCO(f2,h) =  ιpO(f2,h)/κpO(f2,h)
    BCM(f2,h) =  ιpM(f2,h)/κpM(f2,h)

    # selection gradients for helping (eq. S2.8.40)
    SpO(p,z) = (m.f1/dot(u(f2_fn(z),h̄*p),v(f2_fn(z),h̄*p)))*(-ιpO(f2_fn(z),h̄*p)*C + κpO(f2_fn(z),h̄*p)*B(f2_fn(z),h̄*p))
    SpM(p,z) = (m.f1/dot(u(f2_fn(z),h̄*p),v(f2_fn(z),h̄*p)))*(-ιpM(f2_fn(z),h̄*p)*C + κpM(f2_fn(z),h̄*p)*B(f2_fn(z),h̄*p))

    # selection gradients for x and y (eq. S2.8.41)
    Sx(x,y,z) = dpdx(x,y)*SpM(p_fn(x,y),z)
    Sy(x,y,z) = dpdy(x,y)*SpO(p_fn(x,y),z)

    # selection gradients for z (eq. S2.9.6)
    Sz(p,z) = (1/dot(u(f2_fn(z),h̄*p),v(f2_fn(z),h̄*p)))*df2dz(z)*κz(f2_fn(z),h̄*p)*D(f2_fn(z),h̄*p)

    # genetic variance function (eq. S.8.0.1d)
    β = 100
    Gzeta(ζ) = 1-exp(-β*ζ)

    # evolutionary dynamics (eq. S6.1.1 for genetically uncorrelated traits)
    dxdt(x,y,z) = m.Gx*Gzeta(x)*Sx(x,y,z)
    dydt(x,y,z) = m.Gy*Gzeta(y)*Sy(x,y,z)
    dzdt(x,y,z) = m.Gz*Gzeta(z)*Sz(p_fn(x,y),z)

    ## function to be used by the ODE solvers:

    # evolutionary dynamics of x, y, and z
    function evoldyn(dzvec,zvec,p,t)
        x = zvec[1]
        y = zvec[2]
        z = zvec[3]
        dzvec[1] = dxdt(x,y,z)
        dzvec[2] = dydt(x,y,z)
        dzvec[3] = dzdt(x,y,z)
    end

    # evolutionary dynamics of x, y when z does not evolve (i.e., z=z0)
    function evoldyn_z0(dzvec,zvec,p,t)
        x = zvec[1]
        y = zvec[2]
        dzvec[1] = dxdt(x,y,z0)
        dzvec[2] = dydt(x,y,z0)
    end

    # evolutionary dynamics of x, y when z evolves at infinite speed (i.e., z=z*(h))
    function evoldyn_zfast(dzvec,zvec,p,t)
        x = zvec[1]
        y = zvec[2]
        zs = zstar(h̄*p_fn(x,y))
        dzvec[1] = dxdt(x,y,zs)
        dzvec[2] = dydt(x,y,zs)
    end

    return ModelFunctions(sM,f2_fn,h_fn,Π2,zstar,f2star,B,C,D,u,v,p_fn,dpdx,dpdy,BCO,BCM,SpO,SpM,Sx,Sy,Sz,dxdt,dydt,dzdt,evoldyn,evoldyn_z0,evoldyn_zfast)

end

# get main outputs (e.g., for promoters plots)
function get_mainoutputs(sol,mp::ModelParameters,mf::ModelFunctions,evoldyn_type)
    t = sol.t
    x = sol[1,:]
    y = sol[2,:]
    n = length(x)
    p = mf.p.(x,y)
    h = hbar(mp) * p
    if evoldyn_type == "z0" # z does not evolve
        z0 = mf.zstar(0)
        z = z0*ones(n) # set z to initial value
    elseif evoldyn_type == "zfast" # z evolves fast
        z = mf.zstar.(h)  # set z to optimal value
    elseif evoldyn_type == "standard" # z evolves at finite speed
        z = sol[3,:]
    end
    return t,x,y,z,p
end

# get all outputs (for main results)
function get_alloutputs(sol,mp::ModelParameters,mf::ModelFunctions,evoldyn_type)

    (t,x,y,z,p) = get_mainoutputs(sol,mp,mf,evoldyn_type)

    h = mf.h(p)
    f2 = mf.f2.(z)
    sM = mf.sM.(f2,h)
    sM_without_helpers = max.(mf.sM.(f2,0.),0.)
    Π2 = mf.Π2.(f2,h)
    BC = mf.B.(f2,h)/mf.C
    BCO = mf.BCO.(f2,h)
    BCM = mf.BCM.(f2,h)

    return t,x,y,z,p,f2,h,sM,sM_without_helpers,Π2,BC,BCO,BCM

end

# identify benefit-cost zones (e.g., to add background colors to time series)
function get_conflict_flags(p,z,mf::ModelFunctions)

    SpOs = mf.SpO.(p,z)
    SpMs = mf.SpM.(p,z)

    nohelping_flags = ((SpOs .< 0) .& (SpMs .< 0))
    conflict_flags = ((SpOs .< 0) .& (SpMs .> 0))
    voluntary_flags = ((SpOs .> 0) .& (SpMs .> 0))

    return nohelping_flags,conflict_flags,voluntary_flags

end

end
