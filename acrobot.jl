using Plots
using LinearAlgebra
using Random
using JLD2, FileIO

# Global vars
damping = 0.1
gravity = 9.8
dt = 0.01

# Runge-Kutta 4 forward integraor to get next state given control
function rk4(state, u)
    global dt
    massPos = state[1:2]
    massVel = state[3:4]
    l0 = dt * xdot(state, u)[3:4]
    k0 = dt * massVel;
    l1state = vcat(massPos + k0*0.5, massVel + l0*0.5)
    wrapstate!(l1state)
    l1 = dt * xdot(l1state, u)[3:4]
    k1 = dt * (massVel + k0*0.5);
    l2state = vcat(massPos + k1*0.5, massVel + l1*0.5)
    wrapstate!(l2state)
    l2 = dt * xdot(l2state, u)[3:4]
    k2 = dt * (massVel + l1*0.5);
    l3state = vcat(massPos + k2, massVel + l2)
    wrapstate!(l3state)
    l3 = dt * xdot(l3state, u)[3:4]
    k3 = dt * (massVel + l2);
    massPosNext = massPos + 1.0/6 * (k0 + 2*k1 + 2*k2 + k3);
    massVelNext = massVel + 1.0/6 * (l0 + 2*l1 + 2*l2 + l3);
    return vcat(massPosNext, massVelNext)
end

# Convert joint angles into x and y positions of the two masses
function cartcoord(state)
    x1 = cos(state[1]+pi/2);
    y1 = sin(state[1]+pi/2);
    x2 = x1 + cos(state[1]+state[2]+pi/2);
    y2 = y1 + sin(state[1]+state[2]+pi/2);
    return [x1, y1, x2, y2]
end

# Dynamics, calculate accelerations
function xdot(x, u)
    M = vcat(transpose([3 + 2*cos(x[2]), 1+cos(x[2])]), transpose([1+cos(x[2]), 1]))

    # Coriolis, centripetal and graviational forces
    c1 = x[4]*(2*x[3]+x[4])*sin(x[2]) + 2*gravity*sin(x[1]) + gravity*sin(x[1]+x[2])
    c2 = -x[3]^2*sin(x[2]) + gravity*sin(x[1]+x[2])
    
    # passive dynamics
    a = vcat(x[3:4], M \ [c1-damping*x[3], c2-damping*x[4]])

    # control gain; use B = [0;0; inv(M)] to allow control of both joints
    Mback = M\[0,1]
    B = [0,0, Mback[1], Mback[2]]

    xdot = a + B*u

    return xdot
end

# Wrap joint angles to stay within (-pi, pi)
function wrapstate!(state)
    if state[1] > pi
        state[1] -= 2*pi
    end
    if state[1] < -pi
        state[1] += 2*pi
    end
    if state[2] > pi
        state[2] -= 2*pi
    end
    if state[2] < -pi
        state[2] += 2*pi
    end
end

# Rollout a whole trajectory given a starting state and control sequence
function forward(xstart, u)
    T = length(u) + 1
    pos = zeros(4, T)
    pos[:,1] = xstart
    for i = 2:T
        next = rk4(pos[:, i-1], u[i-1])
        wrapstate!(next)
        pos[:, i] = next
    end
    return pos
end

function test()
    gr()
    println("start")
    T = 3
    step = 1e-1
    niter = 100
    u = load("./LQR.jld2", "us")
    # pos = forwardLQR(xinit, u)
    pos = forward(xinit, u)
    println("animating")
    anim = @animate for i = 1:3:Int(T/dt) # Only plot every third frame
        coord = cartcoord(pos[:, i])
        plot([0, coord[1], coord[3]], [0, coord[2], coord[4]], xlim=(-2, 2), ylim=(-2,2))
    end
    gif(anim, "./LQRcheck.gif")
end