#pragma once
#include "core/cpu_vision_port.h"
#include "targets/qwen3_6/impl/vision/vision_host.h"
#include <memory>
namespace ninfer::targets::qwen3_6 {
inline float cpu_half(std::uint16_t x){
 const unsigned sign=(x&0x8000U)<<16;unsigned e=(x>>10)&31,m=x&1023;
 if(e==0){if(!m)return std::bit_cast<float>(sign);int shift=-14;while(!(m&1024)){m<<=1;--shift;}return std::bit_cast<float>(sign|(unsigned(shift+127)<<23)|((m&1023)<<13));}
 if(e==31)return std::bit_cast<float>(sign|0x7f800000U|(m<<13));
 return std::bit_cast<float>(sign|((e+112)<<23)|(m<<13));
}
inline std::vector<float> cpu_decode(const Tensor&t){
 if(!t.is_contiguous()||!vision_host_store().in_blob(t.data))throw std::runtime_error("CPU vision tensor must be contiguous host data");
 std::vector<float> out(static_cast<std::size_t>(t.numel()));
 if(t.dtype==DType::BF16){auto p=static_cast<const std::uint16_t*>(t.data);for(std::size_t i=0;i<out.size();++i)out[i]=std::bit_cast<float>(std::uint32_t(p[i])<<16);}
 else if(t.dtype==DType::FP32){auto p=static_cast<const float*>(t.data);std::copy_n(p,out.size(),out.data());}
 else throw std::runtime_error("CPU vision tensor dtype unsupported");return out;
}
inline std::vector<float> cpu_decode(const Weight&w){
 if(!vision_host_store().in_blob(w.qdata)||w.hadamard_signs||w.hadamard_perm_rep>1||w.n<=0||w.k<=0)throw std::runtime_error("CPU vision requires unrotated host weights");
 if(w.layout!=QuantLayout::RowSplit||w.scale_dtype!=DType::FP16)throw std::runtime_error("CPU vision weight layout unsupported");
 int bits=0;switch(w.qtype){case QType::Q4G64_F16S:bits=4;break;case QType::Q5G64_F16S:bits=5;break;case QType::Q6G64_F16S:bits=6;break;case QType::W8G32_F16S:bits=8;break;default:throw std::runtime_error("CPU vision quantization unsupported");}
 const int group=bits==8?32:64;const int groups=w.padded_shape[1]/group;
 if(w.group!=group||groups<=0||w.padded_shape[1]%group)throw std::runtime_error("CPU vision group mismatch");
 std::vector<float> out(static_cast<std::size_t>(w.n)*w.k);
 const auto* low=static_cast<const std::uint8_t*>(w.qdata);const auto* high=static_cast<const std::uint8_t*>(w.qhigh);const auto* scales=static_cast<const std::uint16_t*>(w.scales);
 for(int row=0;row<w.n;++row)for(int g=0;g<groups;++g){
  const auto offset=static_cast<std::size_t>(row)*groups+g;float scale=cpu_half(scales[offset]);
  const auto* q=low+offset*32;const auto* h=high?high+offset*(bits==5?8:16):nullptr;
  for(int lane=0;lane<group && g*group+lane<w.k;++lane){
   int value;if(bits==8)value=static_cast<std::int8_t>(q[lane]);
   else{unsigned v=(q[lane/2]>>((lane%2)*4))&15;
    if(bits==5)v|=((h[lane/8]>>(lane%8))&1)<<4;
    if(bits==6)v|=((h[lane/4]>>((lane%4)*2))&3)<<4;
    value=(v&(1<<(bits-1)))?int(v)-(1<<bits):int(v);
   }
   out[static_cast<std::size_t>(row)*w.k+g*group+lane]=value*scale;
  }
 }
 return out;
}
inline std::shared_ptr<ninfer_cpu_port::CpuVisionWeights> cpu_vision_weights(const VisionWeights& source){
 std::lock_guard guard(ninfer_cpu_backend::encode_mutex());auto& store=vision_host_store();if(store.cpu_weights)return store.cpu_weights;
 auto p=std::make_shared<ninfer_cpu_port::CpuVisionWeights>();auto& w=*p;
 w.hidden=VisionBackboneConfig::hidden;w.heads=VisionBackboneConfig::heads;w.intermediate=VisionBackboneConfig::intermediate;w.patch_width=VisionBackboneConfig::patch_dim;w.merge_unit=VisionBackboneConfig::merge_unit;w.output_hidden=source.merger_fc2.n;w.position_rows=VisionBackboneConfig::position_embeddings;
 const auto& c=source.common;
 w.patch_embedding=cpu_decode(c.patch_embedding);w.patch_embedding_bias=cpu_decode(c.patch_embedding_bias);w.position_embedding=cpu_decode(c.position_embedding);
 for(const auto& l:c.layers){ninfer_cpu_port::CpuVisionWeights::Layer x;
  x.norm1_weight=cpu_decode(l.norm1_weight);x.norm1_bias=cpu_decode(l.norm1_bias);x.qkv=cpu_decode(l.qkv);x.qkv_bias=cpu_decode(l.qkv_bias);x.output=cpu_decode(l.output);x.output_bias=cpu_decode(l.output_bias);x.norm2_weight=cpu_decode(l.norm2_weight);x.norm2_bias=cpu_decode(l.norm2_bias);x.fc1=cpu_decode(l.fc1);x.fc1_bias=cpu_decode(l.fc1_bias);x.fc2=cpu_decode(l.fc2);x.fc2_bias=cpu_decode(l.fc2_bias);w.layers.push_back(std::move(x));
 }
 w.merger_norm_weight=cpu_decode(c.merger_norm_weight);w.merger_norm_bias=cpu_decode(c.merger_norm_bias);w.merger_fc1=cpu_decode(c.merger_fc1);w.merger_fc1_bias=cpu_decode(c.merger_fc1_bias);w.merger_fc2=cpu_decode(source.merger_fc2);w.merger_fc2_bias=cpu_decode(source.merger_fc2_bias);
 store.cpu_weights=p;std::cerr<<"CPU Vision | host FP32 weights decoded; no GPU tower allocation\n";return p;
}
}
