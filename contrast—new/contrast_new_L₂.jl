using CairoMakie, LinearAlgebra, SparseArrays

# --------------- 全局参数 ---------------
𝑘 = 100
𝑚 = 1.0
q̇₀ = 5.0
q₀ = 1.0
𝜔 = sqrt(𝑘/𝑚)


𝑢(t) = q₀ .* cos.(𝜔 .* t) + q̇₀/𝜔 .* sin.(𝜔 .* t)

# --------------- 2. 第一段代码：FEM方法1（蓝色虚线）---------------
ξ = [-1/3^0.5,1/3^0.5]
𝑤 = [1.0,1.0]

h = [0.1,0.05,0.025,0.0125]
L₂_WR = zeros(4)
L₂_HM = zeros(4)
L₂_MIXHM = zeros(4)
for (j,Δt) in enumerate(h)
t_fem = 0.0:Δt:8.0  
nₚ = length(t_fem)
nₑ = nₚ - 1
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

L₂_Δu = 0.0
L₂_u = 0.0
for i in 1:nₑ
    t₁ = t_fem[i]
    t₂ = t_fem[i+1]
    u₁ = x_fem1[i]
    u₂ = x_fem1[i+1]
    for (ξᵢ,𝑤ᵢ) in zip(ξ,𝑤)
        N₁ = 0.5*(1-ξᵢ)
        N₂ = 0.5*(1+ξᵢ)
        tᵢ = t₁*N₁ + t₂*N₂
        uᵢ = u₁*N₁ + u₂*N₂
        L₂_Δu += (uᵢ - 𝑢(tᵢ))^2
        L₂_u += 𝑢(tᵢ)^2
    end
end
L₂_WR[j] = (L₂_Δu/L₂_u)^0.5


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


# --------------- 4. 第三段代码：Mixed-FEM方法（绿色实线）---------------
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

end

h = log10.(h)
L₂_WR = log10.(L₂_WR)

slop_WR = (L₂_WR[end] - L₂_WR[end-1])/(h[end]- h[end-1])
fig = Figure()
ax = Axis(fig[1, 1]) 
#     xlabel = "T", 
#     ylabel = "x",
#     title = "Exact + FEM1 + FEM2  (Δt=0.075)")

lines!(h,L₂_WR)
fig