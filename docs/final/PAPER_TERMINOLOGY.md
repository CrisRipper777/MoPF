# Frozen paper terminology

## Preferred terms

| Concept | Use this term | Meaning in this project |
|---|---|---|
| problem setting | Multimodal Attributed Graph (MAG) | graph with node attributes from text and visual modalities |
| graph support | physical topology / physical graph | observed edges; unchanged by MGSC |
| attributes | modality-intrinsic semantics | information present in a modality before graph contextualization |
| edge mechanism | modality-specific relation calibration | MRC response and bounded relation weights on physical edges |
| propagated information | structural context | information proposed by physical-neighbor propagation |
| intermediate representation | context state | a modality/order-specific mixture of intrinsic state and neighborhood proposal |
| local message | neighborhood proposal | `N_{i,k}^m` before adaptive formation |
| node/order mechanism | adaptive context gate | `g_{i,k}^m` that forms the context state |
| multi-order representation | multi-order context state bank | `B_i^m=[S_{i,0}^m,...,S_{i,K}^m]` |
| order mixing mechanism | cross-order interaction | Q/K/V interaction over order tokens |
| final order readout | interaction-aware integration | eta-weighted composition of interacted states |
| multimodal output | late multimodal fusion | text and visual paths meet only after independent refinement |

## Terms to avoid

Do not use trajectory, smoothing depth, reliable context, relation reliability, optimal depth, causal gate, or universal module necessity as the main terminology. Do not treat M1 preferred lambda as gate ground truth. Do not interpret semantic similarity as relation usefulness or reliability. Do not describe frozen inference interventions as retrained performance.
