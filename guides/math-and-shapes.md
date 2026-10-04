# 公式、维度与易错点速查

## 统一符号

B=batch；T=context；d=modelwidth；h=Qheads；hkv=KVheads；dh=headwidth；f=FFNwidth；L=layers；V=vocab；N=parameters；D=trainingtokens；p=worldsize。注意D在某些并行题代表modelwidth，读公式前先核对符号。

| 计算 | 输入→输出 | 主成本/检查 |
|---|---|---|
| embedding | IDs[B,T]→X[B,T,d] | Vd参数，lookup |
| QKV | [B,T,d]→[B,h,T,dh] | MHA共3d²参数 |
| score | Q Kᵀ→[B,h,T,T] | 2BT²d FLOPs |
| attention weighted value | P V→[B,h,T,dh] | 2BT²d FLOPs |
| output proj | [B,T,d]→[B,T,d] | d²参数 |
| SwiGLU branches | [B,T,d]→两份[B,T,f]→[B,T,d] | 3df参数，6BTdf前向FLOPs |
| LM head | [B,T,d]→[B,T,V] | Vd参数、2BTdV FLOPs |

untied参数 `N=2Vd+L(4d²+3df+2d)+d`。matmul FLOPs `L(8BTd²+6BTdf+4BT²d)+2BTdV`。训练近似前向3倍；6ND忽略attention等。FP32 Adam静态16Nbytes；KV BF16bytes为 `2BLT hkv dh×2`。

## 数值稳定

softmax减max；CE用logsumexp减正确classlogit；RMSNormFP32reduction，eps在sqrt内；DPO用-logsigmoid；重要性用log-ratio，极端值可溢出应监控，不静默改目标。ratio clip只改surrogate，不保证policyKL硬上界。

## 链式法则与梯度

Y=XW：dX=dY Wᵀ，dW=XᵀdY。softmaxvectorp：J=diag(p)−ppᵀ，反向 `dx=p⊙(dp−Σp⊙dp)`。注意力反向 `dV=PᵀdO,dP=dO Vᵀ,dS=P⊙(dP−row_sum(P⊙dP))`，再对QKᵀ回传。

通过有限差分只用于很小tensor，float64降低噪声；randomupstreamgradient比对output.sum更能发现某些对称错误。浮点顺序变化用合理atol/rtol，不能通过扩大容差掩盖结构性错误。

## 关键统计推导

固定C：`N*∝C^{β/(α+β)}`，`D*∝C^{α/(α+β)}`。IsoFLOPs取离散最小点后拟合；rawloss存在E，logloss不自动线性。

MinHashcollision probability=Jaccard；LSHcandidate=`1−(1−s^r)^b`；候选仍需exactJaccard确认。

BernoulliaccuracySE≈`sqrt(p(1−p)/n)`；同问题多模型比较用配对统计；fewshot/seed/解析不一致不能归为模型差异。

## 后训练对齐

prompt+response分开编码→concatenate→input[:-1]/labels[1:]/mask[1:]。mask在labels对应位置，causal前缀仍参与forward。sequence loss是先每response平均再平均，tokenmean是所有有效token共同平均，两者对长短答案权重不同。

GRPO优势 `(r−groupmean)/(samplestd+eps)`；old与advantage detach；Dr无std用constant；GSPOratio=`exp(maskedmean(logπ−logπold))`；reference与oldpolicy是两个角色。

DPOdelta=`(logπchosen−logπrejected)−(logπrefchosen−logπrefrejected)`；loss=`−logsigmoid(βdelta)`；policy=ref时log2，梯度仍一般非零。

## 性能口径

算术强度=FLOPs/bytes；roofline=min(peak,bw×I)；ringallreduce每rank≈2(p−1)S/p；effectivebatch=micro×accum×DP。MFU分母注明dense/precision/设备数。TTFT、TPOT、p95、goodput各自解释，不用一个tokens/s覆盖服务质量。
