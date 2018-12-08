
function grad_descent(xinit, uinit, T, niter, step)
   u = copy(uinit)
   nu = length(u)
   costs = zeros(niter)
   for i = 1:niter
       pos = forward(xinit, u)
       curr_cost = total_cost(pos, u)
       costs[i] = curr_cost
       println("Iteration: "*string(i)*"\tTotal Cost: "*string(curr_cost))
       grad = zeros(nu)
       for j = 1:nu
           leftU = copy(u)
           leftU[j] -= eps
           rightU = copy(u)
           rightU[j] += eps
           leftPos = forward(xinit, leftU)
           rightPos = forward(xinit, rightU)
           leftcost = total_cost(leftPos, leftU)
           rightcost = total_cost(rightPos, rightU)
           grad[j] = (rightcost - leftcost) / (2*eps)
       end
       println(LinearAlgebra.norm(grad))
       println(maximum(grad))
       println(minimum(grad))
       u = u - step*grad
   end
   return u, costs
end

function SGD(xinit, uinit, T, niter, step)
   u = copy(uinit)
   costs = zeros(niter)
   nu = length(u)
   for i = 1:niter
       sampind = randperm(nu)
       for j = 1:nu
           if sampind[j] != 1
               pos = forward(xinit, u[1:nu-1])
           else
               pos = xinit
           end
           leftU = u[sampind[j]] - eps
           rightU = u[sampind[j]] + eps
           leftPos = rk4(pos[:, end], leftU)
           rightPos = rk4(pos[:, end], rightU)
           leftcost = true_cost(leftPos, leftU)
           rightcost = true_cost(rightPos, rightU)
           grad = (rightcost - leftcost) / (2*eps)
           u[sampind[j]] = u[sampind[j]] + step*grad
       end
       pos = forward(xinit, u)
       curr_cost = total_cost(pos, u)
       costs[i] = curr_cost
       println("Iteration: "*string(i)*"\tTotal Cost: "*string(curr_cost))
   end
   return u, costs
end

function step_search(x, u, curr_cost, grad, step)
   newu = u + step*grad
   rollout = forward(x, newu)
   roll_cost = total_cost(rollout, newu)
   step_incr = step/10
   curr_step = step
   while roll_cost > curr_cost
       curr_step -= step_incr
       newu = u + curr_step*grad
       rollout = forward(x, newu)
       roll_cost = total_cost(rollout, newu)
   end
   return newu
end

function int_lambda(lambda, x, u)
    lam_vel = lambdadot(lambda, x, u)
    return lambda + dt*lam_vel
end

function lambdadot(lambda, x, u)
    return true_lx(x, u) + transpose(finiteDiff(x, u, "x")) * lambda
end

function lx(x, u)
    return k^2*[x[1], x[2], 0, 0]# + k*[1, 1, 0, 0]
end


function cost(x, u)
    return r/2*u^2 + 1/2*k^2*(x[1]^2+x[2]^2)
end

function uGrad!(G, u)
    x = forward(xinit, u)
    lambdas = zeros(4, T+1)
    lambdas[:, T+1] = lx(x[:, T+1], 0)
    for i = T:-1:1
        lambdas[:, i] = int_lambda(lambdas[:, i+1], x[:, i], u[i])
    end
    for i = 1:T
        G[i] = lu(x[:, i], u[i]) + transpose(finiteDiff(x[:, i], u[i], "u")) * lambdas[:, i+1]
    end
end

function accel_trajopt(xinit, niter, max_step)
    # prev_a = 1
    # curr_a = .5*(1+sqrt(4+1))
    curr_a = 1
    prev_u = zeros(Int(T/dt))
    curr_u = zeros(Int(T/dt))
    curr_y = zeros(Int(T/dt))
    costs = zeros(0)
    for i = 1:niter
        curr_step = max_step
        curr_traj = forward2(xinit, curr_u)
        curr_cost = total_cost(curr_traj, curr_u)
        append!(costs, curr_cost)
        println("Iteration: "*string(i)*"\tTotal Cost: "*string(curr_cost))
        next_a = .5*(1+sqrt(4*curr_a^2+1))
        t = (curr_a-1) / next_a
        y = (1+t)*curr_u-t*prev_u
        pos = forward2(xinit, y)
        grads = getgrads(pos, y)
        # next_y = curr_u + max_step*grads
        newu = y + curr_step*grads
        do_adapt = false
        if do_adapt
            rollout = forward2(xinit, newu)
            roll_cost = total_cost(rollout, newu)
            while (roll_cost > curr_cost)# && curr_step > 1e-6)
                curr_step = curr_step / 2
                newu = y + curr_step*grads
                rollout = forward2(xinit, newu)
                roll_cost = total_cost(rollout, newu)
            end
        end
        # println(curr_step)
        u = newu
        # new_u = y + max_step*grads
        curr_a = next_a
        prev_u = curr_u
        curr_u = newu
    end
    return curr_u, costs
end