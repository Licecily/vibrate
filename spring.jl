using CairoMakie, LinearAlgebra, SparseArrays

𝑘 = 100
𝑚 = 1.0
q̇₀ = 5.0
q₀ = 1.0

fig = Figure()
# Axis(fig[1, 1])
ax = Axis(fig[1, 1], xlabel = "T", ylabel = "x",title = "Hamilton vs Exact Solution")
𝑡 = 0.0:0.01:8.0
𝜔 = (𝑘/𝑚)^0.5
𝑥 = q₀.*cos.(𝜔.*𝑡) + q̇₀/𝜔.*sin.(𝜔.*𝑡)
u(t) = q₀*cos(𝜔*t) + q̇₀/𝜔*sin(𝜔*t)
lines!(ax, 𝑡, 𝑥, color = :black, linewidth=2, label="Exact Solution")

# lines!(𝑡, 𝑝, color = :black)

t = 0.0:0.01:8.0
nₚ = length(t)
nₑ = nₚ-1

# FEM weak LM
k_uu = spzeros(nₚ, nₚ)  
k_uv = spzeros(nₚ, nₚ)
k_vv = spzeros(nₚ, nₚ)
f_u = zeros(nₚ)
f_v = zeros(nₚ)

for i in 1:nₑ
    t₁ = t[i]
    t₂ = t[i+1]
    𝐿 = t₂ - t₁

    k_uu[i,i] += -𝑘*0.5
    k_uu[i,i+1] += 𝑘*0.5
    k_uu[i+1,i] += -𝑘*0.5
    k_uu[i+1,i+1] += 𝑘*0.5
    
    k_vv[i,i] += -𝑚*0.5
    k_vv[i,i+1] += 𝑚*0.5
    k_vv[i+1,i] += -𝑚*0.5
    k_vv[i+1,i+1] += 𝑚*0.5

    k_uv[i,i] += -𝐿*𝑘/3
    k_uv[i,i+1] += -𝐿*𝑘/6
    k_uv[i+1,i] += -𝐿*𝑘/6
    k_uv[i+1,i+1] += -𝐿*𝑘/3
end

α = 1e9   
β = 1e9   

k_uu[1, 1] += α   
f_u[1] = α * q₀   

k_vv[1, 1] += β   
f_v[1] = β * q̇₀  


k = [k_uu   k_uv    ;
     -k_uv'  k_vv   ]
f = [f_u; f_v]


d = k \ f

# val = eigvals(k_uu)
# val = eigvals(k_vv)
# val = eigvals(k_uv)
# val = eigvals(k_uu*inv(k_vv)*k_uv')
e = d[1:nₚ] - 𝑥
# lines!(ax, t, e, color = :red)
lines!(ax, t, d[1:nₚ], color = :blue, linewidth=2, label="FEM Solution", linestyle=:dash)
axislegend(ax, position = :rt)  # rt = 右上角
# lines!(t, d[nₚ+1:end-1], color = :blue)

# invisible_line = lines!(ax, [0, 0], [0, 0], color = :white, label="Δt=0.05", visible=false)
# blue_line = lines!(ax, t[1:2], q[1:2], color = :blue, label="Hamilton")
# black_line = lines!(ax, t[1:2], 𝑢.(t[1:2]), color = :black, label="Exact Solution")
# red_line = lines!(ax, t[1:2], e[1:2], color = :red, label="error")
# leg = Legend(fig[1, 2], [red_line, invisible_line], ["error", "Δt=0.05"], position=(0.95, 0.95))
# leg = Legend(fig[1, 2], [blue_line, black_line, invisible_line], ["Hamilton", "Exact Solution", "Δt=0.05"], position=(0.95, 0.95))

fig

save("hmd_uv_0.01.png", fig)