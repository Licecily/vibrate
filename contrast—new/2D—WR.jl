using CairoMakie, LinearAlgebra, SparseArrays
# ================ 全局参数（和之前完全一致） ================
𝑘 = 100
𝑚 = 1.0
q̇₀ = 5.0
q₀ = 1.0
𝜔 = sqrt(𝑘/𝑚)
Δt = 0.075
t_fem = 0.0:Δt:8.0  
nₚ = length(t_fem)
nₑ = nₚ - 1

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = "T", 
    ylabel = "x",
    title = "Exact + FEM1 + FEM2 + Mixed-FEM + 2D WR FEM (Δt=0.075)")

# ================ 1. 精确解（和之前一致） ================
t_exact = 0.0:0.001:8.0
x_exact = q₀ .* cos.(𝜔 .* t_exact) + q̇₀/𝜔 .* sin.(𝜔 .* t_exact)
lines!(ax, t_exact, x_exact, color = :black, linewidth=2, label="Exact Solution")

# ================ 2. WR FEM（蓝色实线，和之前一致） ================
k_uu = spzeros(nₚ, nₚ)  
k_uv = spzeros(nₚ, nₚ)
k_vv = spzeros(nₚ, nₚ)
f_u = zeros(nₚ)
f_v = zeros(nₚ)

for i in 1:nₑ
    t₁ = t_fem[i]
    t₂ = t_fem[i+1]
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
kᵅ = spzeros(nₚ, nₚ)  
kᵅ[1, 1] = α         
fᵅ = zeros(nₚ)
fᵅ[1] = α * q₀ 

kᵝ = spzeros(nₚ, nₚ)
kᵝ[1, 1] = α         
fᵝ = zeros(nₚ)
fᵝ[1] = α * q̇₀ 

K1 = [k_uu+kᵅ   k_uv    ;
     -k_uv'  k_vv+kᵝ   ]
f1 = [f_u+fᵅ; f_v+fᵝ]
d1 = K1 \ f1
x_fem1 = d1[1:nₚ]

lines!(ax, t_fem, x_fem1, color = :blue, linewidth=2, 
       linestyle=:solid, label="WR FEM ")

# ================ 3. Hamiltonian FEM（红色实线，和之前一致） ================
k = zeros(nₚ,nₚ)
f = zeros(nₚ)

for i in 1:nₑ
    t₁ = t_fem[i]
    t₂ = t_fem[i+1]
    𝐿 = t₂ - t₁
    k[i,i] += 𝑚/𝐿 - 𝑘/3*𝐿
    k[i,i+1] += -𝑚/𝐿 - 𝑘/6*𝐿
    k[i+1,i] += -𝑚/𝐿 - 𝑘/6*𝐿
    k[i+1,i+1] += 𝑚/𝐿 - 𝑘/3*𝐿
end

𝑃₀ = 𝑚*q̇₀
f[1] -= 𝑃₀

α = 1e9   
kᵅ = zeros(nₚ,nₚ)
fᵅ = zeros(nₚ)
kᵅ[1,1] += α
fᵅ[1] += α*q₀
kᵝ = zeros(nₚ,nₚ)
fᵝ = zeros(nₚ)
kᵝ[nₚ,nₚ] += α
d2 = [k+kᵅ -k;-k kᵝ]\[fᵅ;-f+fᵝ]
x_fem2 = d2[1:nₚ]

lines!(ax, t_fem, x_fem2, color = :red, linewidth=2, label="Hamiltonian Method")

# ================ 4. Mixed-FEM（绿色实线，和之前一致） ================
kᵤᵤ = zeros(nₚ,nₚ)
kₚₚ = zeros(nₚ,nₚ)
kᵤₚ = zeros(nₚ,nₚ)
f = zeros(nₚ)

for i in 1:nₑ
    t₁ = t_fem[i]
    t₂ = t_fem[i+1]
    𝐿 = t₂ - t₁

    kᵤᵤ[i,i] += 𝐿*𝑘/3
    kᵤᵤ[i,i+1] += 𝐿*𝑘/6
    kᵤᵤ[i+1,i] += 𝐿*𝑘/6
    kᵤᵤ[i+1,i+1] += 𝐿*𝑘/3
    
    kₚₚ[i,i] += 𝐿/𝑚/3
    kₚₚ[i,i+1] += 𝐿/𝑚/6
    kₚₚ[i+1,i] += 𝐿/𝑚/6
    kₚₚ[i+1,i+1] += 𝐿/𝑚/3

    kᵤₚ[i,i] -= -0.5
    kᵤₚ[i,i+1] -= -0.5
    kᵤₚ[i+1,i] -= 0.5
    kᵤₚ[i+1,i+1] -= 0.5
end

𝑃₀ = 𝑚*q̇₀
f[1] += 𝑃₀

k = kᵤᵤ - kᵤₚ*(kₚₚ\kᵤₚ')

α = 1e9   
kᵅ = zeros(nₚ,nₚ)
fᵅ = zeros(nₚ)
kᵅ[1,1] += α
fᵅ[1] += α*q₀
kᵝ = zeros(nₚ,nₚ)
fᵝ = zeros(nₚ)
kᵝ[nₚ,nₚ] += α
d3 = [k+kᵅ -k;-k kᵝ]\[fᵅ;-f+fᵝ]
x_fem3 = d3[1:nₚ]

lines!(ax, t_fem, x_fem3, color = :green, linewidth=2, label="Mixed-Hamiltonian Method")

# ================ 5. 二维WR FEM ================
nₓ = 2            
nₜ = nₚ           
c² = 𝑘/𝑚         
α= 1e9  

x = [0.0, 1.0]
t = collect(t_fem)
n = nₓ * nₜ
coords = [(x[i], t[j]) for j in 1:nₜ for i in 1:nₓ]

elements = Vector{NTuple{3,Int}}()
for j in 1:nₜ-1
    n1 = (j-1)*nₓ + 1
    n2 = (j-1)*nₓ + 2
    n3 = j*nₓ + 1
    n4 = j*nₓ + 2
    push!(elements, (n1, n2, n3))
    push!(elements, (n2, n4, n3))
end

Kᵘᵘ = spzeros(n, n)
Kᵘᵛ = spzeros(n, n)
Kᵛᵘ = spzeros(n, n)
Kᵛᵛ = spzeros(n, n)

for (n1,n2,n3) in elements
    x1,t1 = coords[n1]
    x2,t2 = coords[n2]
    x3,t3 = coords[n3]
    A = 0.5 * abs((x2-x1)*(t3-t1) - (x3-x1)*(t2-t1))
    if A == 0
        continue
    end

  dNdx = zeros(3)
    dNdx[1] = (t2 - t3) / (2A)
    dNdx[2] = (t3 - t1) / (2A)
    dNdx[3] = (t1 - t2) / (2A)

    dNdt = zeros(3)
    dNdt[1] = (x3 - x2) / (2A)
    dNdt[2] = (x1 - x3) / (2A)
    dNdt[3] = (x2 - x1) / (2A)

    M = [A/6 A/12 A/12; A/12 A/6 A/12; A/12 A/12 A/6]
    C = (A/3) * dNdt

    for a in 1:3, b in 1:3
        idx_a = a == 1 ? n1 : a == 2 ? n2 : n3
        idx_b = b == 1 ? n1 : b == 2 ? n2 : n3
        Kᵘᵘ[idx_a, idx_b] += 𝑘 * C[b]
        Kᵘᵛ[idx_a, idx_b] += -𝑘 * M[a,b]
        Kᵛᵘ[idx_a, idx_b] += c² * M[a,b]
        Kᵛᵛ[idx_a, idx_b] += C[b]
    end
end

Kᵅ = spzeros(n, n)
Kᵝ = spzeros(n, n)
Fᵅ = zeros(n)
Fᵝ = zeros(n)
init_nodes = 1:nₓ
for node in init_nodes
    Kᵅ[node, node] = α
    Kᵝ[node, node] = α
    Fᵅ[node] = α * q₀
    Fᵝ[node] = α * q̇₀
end

K = [Kᵘᵘ + Kᵅ    Kᵘᵛ        ;
           Kᵛᵘ          Kᵛᵛ + Kᵝ  ]
F = [Fᵅ; Fᵝ]
d4 = K \ F

x_fem4 = [d4[(j-1)*nₓ + 1] for j in 1:nₜ]
lines!(ax, t_fem, x_fem4, color = :purple, linewidth=2, label="2D WR FEM")

# ================ 绘图设置（和之前一致） ================
T = 2π/𝜔  
xlims!(ax, 0, 3*T)
ylims!(ax, -2, 2)

fig
save("./contrast—new/2D_WR_$(string(Δt)).png", fig)