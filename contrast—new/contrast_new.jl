using CairoMakie, LinearAlgebra, SparseArrays

# --------------- 全局参数 ---------------
𝑘 = 100
𝑚 = 1.0
q̇₀ = 5.0
q₀ = 1.0
𝜔 = sqrt(𝑘/𝑚)
t_fem = 0.0:0.075:8.0  
nₚ = length(t_fem)
nₑ = nₚ - 1

fig = Figure()
ax = Axis(fig[1, 1], 
    xlabel = "T", 
    ylabel = "x",
    title = "Exact + FEM1 + FEM2  (Δt=0.075)")

# --------------- 1. 精确解（平滑黑色曲线）---------------
# 用极密网格画完美平滑曲线
t_exact = 0.0:0.001:8.0
x_exact = q₀ .* cos.(𝜔 .* t_exact) + q̇₀/𝜔 .* sin.(𝜔 .* t_exact)
lines!(ax, t_exact, x_exact, color = :black, linewidth=2, label="Exact Solution")

# --------------- 2. 第一段代码：FEM方法1（蓝色虚线）---------------
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

# 绘制FEM1（蓝色虚线）
lines!(ax, t_fem, x_fem1, color = :blue, linewidth=2, 
       linestyle=:dash, label="WR FEM ")

# --------------- 3. 第二段代码：FEM方法2（红色实线）---------------
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
# kᵝ[1,1] += α
kᵝ[nₚ,nₚ] += α
d2 = [k+kᵅ -k;-k kᵝ]\[fᵅ;-f+fᵝ]
x_fem2 = d2[1:nₚ]

# 绘制FEM2（红色实线）
lines!(ax, t_fem, x_fem2, color = :red, linewidth=2, label="Hamiltonian Method")


T = 2π/𝜔  
xlims!(ax, 0, 3*T)  # 限制X轴为2个周期
ylims!(ax, -2, 2) # 适配幅值
axislegend(ax, position = :rt, labelsize=8) 

# 保存并显示
fig
save("contrast_new_0.075.png", fig)