function rk4_2(state, u)
    massPos = state[1:2]
    massVel = state[3:4]
    l0 = dt * xdot2(state, u)[3:4]
    k0 = dt * massVel;
    l1state = vcat(massPos + k0*0.5, massVel + l0*0.5)
    wrapstate!(l1state)
    l1 = dt * xdot2(l1state, u)[3:4]
    k1 = dt * (massVel + k0*0.5);
    l2state = vcat(massPos + k1*0.5, massVel + l1*0.5)
    wrapstate!(l2state)
    l2 = dt * xdot2(l2state, u)[3:4]
    k2 = dt * (massVel + l1*0.5);
    l3state = vcat(massPos + k2, massVel + l2)
    wrapstate!(l3state)
    l3 = dt * xdot2(l3state, u)[3:4]
    k3 = dt * (massVel + l2);
    massPosNext = massPos + 1.0/6 * (k0 + 2*k1 + 2*k2 + k3);
    massVelNext = massVel + 1.0/6 * (l0 + 2*l1 + 2*l2 + l3);
    return vcat(massPosNext, massVelNext)
end

function rk4_lin(state, u)
    massPos = state[1:2]
    massVel = state[3:4]
    l0 = dt * xdotlin(state, u)[3:4]
    k0 = dt * massVel;
    l1state = vcat(massPos + k0*0.5, massVel + l0*0.5)
    wrapstate!(l1state)
    l1 = dt * xdotlin(l1state, u)[3:4]
    k1 = dt * (massVel + k0*0.5);
    l2state = vcat(massPos + k1*0.5, massVel + l1*0.5)
    wrapstate!(l2state)
    l2 = dt * xdotlin(l2state, u)[3:4]
    k2 = dt * (massVel + l1*0.5);
    l3state = vcat(massPos + k2, massVel + l2)
    wrapstate!(l3state)
    l3 = dt * xdotlin(l3state, u)[3:4]
    k3 = dt * (massVel + l2);
    massPosNext = massPos + 1.0/6 * (k0 + 2*k1 + 2*k2 + k3);
    massVelNext = massVel + 1.0/6 * (l0 + 2*l1 + 2*l2 + l3);
    return vcat(massPosNext, massVelNext)
end

function euler(state, u)
    curr_xdot = xdot(state, u)
    return state + dt*curr_xdot
end

function cartcoord2(state)
    x1 = sin(state[1]);
    y1 = cos(state[1]);
    x2 = x1 + sin(state[2]);
    y2 = y1 + cos(state[2]);
    return [x1, y1, x2, y2]
end

# Different dynamics, should allow fo actuation as both joints
function xdot2(x, u)
    diff = x[1] - x[2]
    t2ddot = (u+.5*(x[4]^2*sin(diff)*cos(diff)-2*gravity*sin(x[1])*cos(diff))
                + x[3]^2*sin(diff)+gravity*sin(x[2])) / (1-.5*cos(diff)^2)
    t1ddot = -.5*(t2ddot*cos(diff)+x[3]^2*sin(diff)-2*gravity*sin(x[1]))
    return [x[3], x[4], t1ddot, t2ddot]
    # t1ddot = (u-x[3]^2*sin(diff)*cos(diff)-gravity*sin(x[2])*cos(diff)
    #             -x[4]^2*sin(diff)+2*gravity*sin(x[1])) / (2-cos(diff)^2)
    # t2ddot = -t1ddot*cos(diff)+x[3]^2*sin(diff)+gravity*sin(x[2])
    # return [x[3], x[4], t1ddot, t2ddot]
end

# Linearized dynamics
function xdotlin(x, u)
    A = hcat([0, 0, 1, 0], [0, 0, 0, 1], [gravity, -gravity, 0, 0], [-gravity, 3*gravity, 0, 0])
    A = transpose(A)
    B = [0, 0, -2, 5]
    return A*x + B*u
end

function clampc!(ctrl)
    clamp!(ctrl, -50, 50)
end

function forwardLQR(xstart, u)
    T = length(u) + 1
    pos = zeros(4, T)
    pos[:,1] = xstart
    G = [421.7063, 140.6532, 173.1010, 65.7230]
    us = zeros(T-1)
    for i = 2:T
        uLQR = transpose(G)*pos[:, i-1]
        us[i-1] = uLQR
        next = rk4(pos[:, i-1], us[i-1])
        wrapstate!(next)
        pos[:, i] = next
    end
    save("LQR.jld2", "us", us, "pos", pos)
    return pos
end

function forward2(xinit, u)
    T = size(u)[1] + 1
    pos = zeros(4, T)
    pos[:,1] = xinit
    for i = 2:T
        next = rk4_lin(pos[:, i-1], u[i-1])
        wrapstate!(next)
        pos[:, i] = next
    end
    return pos
end

function forwardEuler(xinit, u)
    T = length(u) + 1
    pos = zeros(4, T)
    pos[:,1] = xinit
    G = [421.7063, 140.6532, 173.1010, 65.7230]
    for i = 2:T
        # uLQR = transpose(G)*pos[:, i-1]
        next = euler(pos[:, i-1], u[i-1])
        wrapstate!(next)
        pos[:, i] = next
    end
    return pos
end