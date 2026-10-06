"""Local DSH front door. Separate home; no global settings or permission overrides."""
from pathlib import Path
import os,sys,re,subprocess,json,shutil
KIT=Path(__file__).resolve().parent.parent
MODULES=Path(os.environ.get('DSH_MODULES_DIR', str(KIT/'dsh-runtime/node_modules/@deepseek-ai')))
node=os.environ.get('DSH_NODE_EXE') or shutil.which('node')
if not node: sys.exit('Node.js not found. Install Node.js or set DSH_NODE_EXE.')
NODE=Path(node)
if not (MODULES/'dsh/lib/bin.js').is_file(): sys.exit('DSH modules not found. Set DSH_MODULES_DIR to your installed @deepseek-ai directory.')
preset=sys.argv[1] if len(sys.argv)>1 else 'minimal'
assert preset in ('minimal','standard')
c={}
for name in ['设置.ini','settings.ini']:
 p=KIT/name
 if p.exists():
  for line in p.read_text(encoding='utf-8-sig').splitlines():
   if '=' in line and not line.lstrip().startswith((';','#')):
    k,v=line.split('=',1);c[k.strip().upper()]=re.split(r'\s+[;#]',v)[0].strip()
mode=(KIT/'current-mode.txt').read_text(encoding='utf-8-sig').strip() if (KIT/'current-mode.txt').exists() else c.get('MODE','kvmem')
if 'iq3xxs' in KIT.name or ('8g' in KIT.name and mode=='kvrk4'):ctx=131072
elif 'swift15' in KIT.name or 'iq2s' in KIT.name:ctx=min(int(c.get('CTX','204800')),131072 if c.get('KV')=='rk8v4' else 262144)
else:ctx=int(c.get({'normal':'NORMAL_CTX','rk8v4':'RK8V4_CTX','kvmem':'KVMEM_CTX','kvrk':'KVRK_CTX','kvrk4':'KVRK4_CTX'}.get(mode,'KVMEM_CTX'),'262144'))
ceiling=32768 if ctx>=262144 else 16384
port=int(c.get('PORT','8084'));model=c.get('MODEL_ID','swift15-iq3xxs' if 'iq3xxs' in KIT.name else 'qwen3.8-27b')
assert 1<=port<=65535 and re.fullmatch(r'[\w./:-]+',model)
web=(MODULES/'dsh-web-app/cordis.patch.yml').read_text(encoding='utf-8')
part=web[web.index('- id: tool-plugin-manager',web.index('# ── the agent plane')):];part=part[:part.index('- insert:')]
patch=part+'\n- id: session-title-llm\n  disabled: true\n- id: deepseek-account\n  disabled: true\n'
patch+='- id: agent-default-model\n  config:\n    provider: oneclick-local\n    model: '+model+'\n'
patch+='- id: llm-pi-ai\n  config:\n    providers:\n      oneclick-local:\n        apiKeyEnv: DSH_ONECLICK_LOCAL_KEY\n        api: openai-completions\n        baseURL: http://127.0.0.1:'+str(port)+'/v1\n        compat:\n          thinkingFormat: deepseek\n        retryPolicy:\n          mode: normal\n          maxRetries: 0\n        models:\n          - id: '+model+'\n            name: Local oneclick\n            contextWindow: '+str(ctx)+'\n            maxTokens: '+str(ceiling)+'\n            reasoning: false\n'
patch+="- insert:\n    - id: tool-subagent-model-selection-settings\n      name: '@deepseek-ai/dsh-tool-subagent/model-selection-settings'\n    - id: agent-preset-registry\n      name: '@deepseek-ai/dsh-agent-preset-registry'\n      config:\n        default: "+preset+'\n'
patch+=(KIT/'DSH-128K-standard.preset.yml' if preset=='standard' else MODULES/'dsh-web-app/presets/minimal.patch.yml').read_text(encoding='utf-8')
home=KIT/'dsh-local';home.mkdir(exist_ok=True);path=home/(preset+'.patch.yml');path.write_text(patch,encoding='utf-8')
profile_dir=home/'home/profiles/oneclick-local';profile_dir.mkdir(parents=True,exist_ok=True)
profile_files={'package.json':json.dumps({'name':'dsh-profile-oneclick-local','private':True,'dependencies':{},'dsh':{'profile':{'bundles':['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless']}}},indent=2),'cordis.yml':'[]\n','cordis.patch.yml':'[]\n'}
for name,content in profile_files.items():
 target=profile_dir/name
 if not target.exists():target.write_text(content,encoding='utf-8')
if '--check-only' in sys.argv:
 print(json.dumps({'preset':preset,'context':ctx,'maxTokens':ceiling,'profile':str(path),'global_config_modified':False}));sys.exit()
env=os.environ.copy();env['DSH_HOME']=str(home/'home');env['DSH_TELEMETRY_DISABLED']='1';env['DSH_ONECLICK_LOCAL_KEY']=env.get('DSH_ONECLICK_LOCAL_KEY') or c.get('API_KEY') or 'local'
env['PATH']=str(NODE.parent)+';'+env.get('PATH','')
# Do not grant unrestricted tool permissions; use DSH defaults/user policy.
extra=sys.argv[2:]
if not extra:
 try:task=input('DSH local task (Ctrl+C to cancel): ').strip()
 except (EOFError,KeyboardInterrupt):sys.exit(0)
 if not task:sys.exit(0)
 extra=['--json',task]
sys.exit(subprocess.call([str(NODE),str(MODULES/'dsh/lib/bin.js'),'--profile','oneclick-local','--patch',str(path),*extra],env=env,cwd=KIT))
