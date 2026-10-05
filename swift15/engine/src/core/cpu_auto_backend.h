#pragma once
// CPU-only vision SGEMM backend and bounded, cached thread calibration.
// Missing/incompatible library falls back to the reference CPU implementation; never GPU.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#endif
namespace ninfer_cpu_backend {
inline unsigned available_threads(){
 unsigned n=std::max(1U,std::thread::hardware_concurrency());
#ifdef _WIN32
 DWORD_PTR process=0,system=0;
 if(GetProcessAffinityMask(GetCurrentProcess(),&process,&system)&&process)n=std::min(n,static_cast<unsigned>(std::popcount(static_cast<std::uint64_t>(process))));
#endif
 return std::clamp(n,1U,64U);
}
inline std::string hardware_key(){
 std::ostringstream s;s<<"cpu-vision-auto-v1/"<<available_threads();
#ifdef _WIN32
 int info[4]{};__cpuid(info,0);s<<'/'<<info[1]<<'/'<<info[2]<<'/'<<info[3];
 __cpuid(info,1);s<<'/'<<info[0];
 for(unsigned i=0x80000002;i<=0x80000004;++i){__cpuid(info,i);for(int j:info)s<<'/'<<j;}
 DWORD_PTR p=0,a=0;GetProcessAffinityMask(GetCurrentProcess(),&p,&a);s<<'/'<<p;
#endif
 return s.str();
}
inline bool compatible_cpu(){
#ifdef _WIN32
 int x[4]{};__cpuid(x,1);
 if(!(x[2]&(1<<27))||!(x[2]&(1<<28)))return false;
 if((_xgetbv(0)&6)!=6)return false;
 __cpuidex(x,7,0);return (x[1]&(1<<5))!=0;
#else
 return false; // portable reference CPU path, no unverified dynamic library ABI
#endif
}
inline std::uint64_t digest(const std::string&s){std::uint64_t h=14695981039346656037ULL;for(unsigned char c:s){h^=c;h*=1099511628211ULL;}return h;}
struct Library {
 using Gemm=void(*)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
 using Threads=void(*)(int);using Text=const char*(*)();
 Gemm gemm=nullptr;Threads threads=nullptr;std::string identity="reference";
 Library(){
  const char* mode=std::getenv("NINFER_CPU_GEMM");
  if((mode&&std::string(mode)=="reference")||!compatible_cpu())return;
#ifdef _WIN32
  std::filesystem::path path;
  if(const char* p=std::getenv("NINFER_CPU_BLAS_DLL"))path=p;
  else{wchar_t exe[32768]{};DWORD n=GetModuleFileNameW(nullptr,exe,32768);if(!n||n>=32768)return;path=std::filesystem::path(exe).parent_path()/"cpu-openblas.dll";}
  HMODULE h=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if(!h){std::cerr<<"CPU Vision | BLAS unavailable; reference CPU fallback\n";return;}
  gemm=reinterpret_cast<Gemm>(GetProcAddress(h,"scipy_cblas_sgemm"));
  threads=reinterpret_cast<Threads>(GetProcAddress(h,"scipy_openblas_set_num_threads"));
  auto config=reinterpret_cast<Text>(GetProcAddress(h,"scipy_openblas_get_config"));
  auto core=reinterpret_cast<Text>(GetProcAddress(h,"scipy_openblas_get_corename"));
  if(!gemm||!threads||!config||!core){gemm=nullptr;threads=nullptr;FreeLibrary(h);std::cerr<<"CPU Vision | incompatible BLAS ABI; reference CPU fallback\n";return;}
  identity=std::string(config())+"/"+core();
  std::error_code e;identity+="/"+std::to_string(std::filesystem::file_size(path,e));
  identity+="/"+std::to_string(std::filesystem::last_write_time(path,e).time_since_epoch().count());
  // Library remains loaded until process exit; entry points cannot dangle.
#endif
 }
};
inline Library& library(){static Library l;return l;}
inline std::mutex& encode_mutex(){static std::mutex m;return m;}
inline std::filesystem::path cache_path(std::uint64_t hash){
 std::filesystem::path p;
 if(const char* custom=std::getenv("NINFER_CPU_CACHE_DIR"))p=custom;
#ifdef _WIN32
 else if(const char* local=std::getenv("LOCALAPPDATA"))p=std::filesystem::path(local)/"NInfer"/"cpu-vision";
#endif
 if(p.empty())return {};
 std::ostringstream name;name<<std::hex<<hash<<".txt";return p/name.str();
}
// Caller holds encode_mutex. Uses a representative real-weight FP32 GEMM, not a full model run.
inline unsigned select_threads(const std::vector<float>& w,int n,int k,int patches){
 const unsigned limit=available_threads();
 if(const char* manual=std::getenv("NINFER_CPU_THREADS")){
  char* end=nullptr;long v=std::strtol(manual,&end,10);
  if(end!=manual&&*end=='\0'&&v>0){unsigned t=static_cast<unsigned>(std::min<long>(v,limit));if(library().gemm)library().threads(t);std::cerr<<"CPU Vision | threads="<<t<<" manual\n";return t;}
 }
 auto& lib=library();if(!lib.gemm){std::cerr<<"CPU Vision | reference CPU threads="<<limit<<"\n";return limit;}
 const int tokens=patches<=256?128:256;
 const auto key=digest(hardware_key()+"/"+lib.identity+"/"+std::to_string(n)+"/"+std::to_string(k)+"/"+std::to_string(tokens));
 static std::map<std::uint64_t,unsigned> memory_cache;
 if(auto it=memory_cache.find(key);it!=memory_cache.end()){lib.threads(it->second);return it->second;}
 const auto path=cache_path(key);unsigned chosen=0;
 if(!path.empty()){
  std::ifstream in(path);std::uint64_t stored=0;unsigned t=0;in>>stored>>t;
  if(in&&stored==key&&t>=1&&t<=limit){chosen=t;std::cerr<<"CPU Vision | threads="<<t<<" cache-hit\n";}
 }
 if(!chosen){
  std::vector<unsigned> candidates{1,2,4,std::max(1U,limit/2),limit};
  for(auto& t:candidates)t=std::min(t,limit);
  std::sort(candidates.begin(),candidates.end());candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());
  std::vector<float> x(static_cast<std::size_t>(k)*tokens),y(static_cast<std::size_t>(n)*tokens);
  for(std::size_t i=0;i<x.size();++i)x[i]=static_cast<float>(static_cast<int>(i%31)-15)/32.0F;
  double best=1e100;
  for(unsigned t:candidates){
   lib.threads(t);lib.gemm(101,111,111,n,tokens,k,1,w.data(),k,x.data(),tokens,0,y.data(),tokens);
   double times[3]{};
   for(double& seconds:times){auto start=std::chrono::steady_clock::now();lib.gemm(101,111,111,n,tokens,k,1,w.data(),k,x.data(),tokens,0,y.data(),tokens);seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
   std::sort(times,times+3);double seconds=times[1];
   std::cerr<<"CPU Vision | calibrate threads="<<t<<" seconds="<<seconds<<'\n';
   if(seconds<best*0.95){best=seconds;chosen=t;}
  }
  if(!chosen)chosen=limit;
  if(!path.empty()){
   std::error_code e;std::filesystem::create_directories(path.parent_path(),e);
   auto tmp=path;tmp+=".tmp.";
#ifdef _WIN32
   tmp+=std::to_string(GetCurrentProcessId());
#else
   tmp+="local";
#endif
   {std::ofstream out(tmp);out<<key<<' '<<chosen<<'\n';}
#ifdef _WIN32
   if(!MoveFileExW(tmp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))std::filesystem::remove(tmp,e);
#else
   std::filesystem::rename(tmp,path,e);if(e)std::filesystem::remove(tmp,e);
#endif
  }
  std::cerr<<"CPU Vision | threads="<<chosen<<" calibrated (representative GEMM)\n";
 }
 memory_cache[key]=chosen;lib.threads(chosen);return chosen;
}
inline bool linear(const std::vector<float>& w,const std::vector<float>& bias,int n,int k,int tokens,const float* x,float* y){
 auto& lib=library();if(!lib.gemm)return false;
 lib.gemm(101,111,111,n,tokens,k,1,w.data(),k,x,tokens,0,y,tokens);
 for(int row=0;row<n;++row)for(int t=0;t<tokens;++t)y[static_cast<std::size_t>(row)*tokens+t]+=bias[row];
 return true;
}
} // namespace ninfer_cpu_backend
