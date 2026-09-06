/* No external scripts, chart CDNs, or client-side credentials. */
const $ = id => document.getElementById(id);
const fmt = value => Number.isFinite(value) ? value.toFixed(4) : '—';
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const phaseNames = {queued:'正在准备',initializing:'初始化研究环境',proposal:'提出新的因子',representation:'刻画因子行为',surrogate:'学习已验证的结果',acquisition:'分配评估预算',train:'验证 Train 表现',validation:'筛选 Validation Top-5',test:'评估 Test 泛化表现',commit:'保存已完成的轮次',checkpoint:'新的阶段结果已就绪',replay:'展开历史发现',resuming:'恢复已保存的搜索',paused:'已安全暂停',completed:'本次运行完成',failed:'运行需要处理',interrupted:'等待恢复'};
const statusNames = {queued:'准备中',running:'运行中',paused:'已暂停',completed:'已完成',failed:'运行失败',interrupted:'已中断'};
let mode='mock', current=null, data=null, archives=[], busy=false, jobs=[], pollBusy=false, sessionBusy=false, pollErrors=0;
let renderKey='';

async function api(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  const body = await response.json();
  if (!response.ok) {
    const detail=body.detail;
    throw new Error(typeof detail==='string' ? detail : detail?.message ? `${detail.message}: ${(detail.missing||[]).join(', ')}` : JSON.stringify(detail));
  }
  return body;
}
function alertMessage(message) { $('alert').hidden=!message; $('alert').textContent=message||''; }
function toast(message) { $('toast').textContent=message; $('toast').hidden=false; setTimeout(()=>$('toast').hidden=true,2200); }
function duration(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  return seconds >= 60 ? `${Math.floor(seconds/60)}m ${Math.round(seconds%60)}s` : `${seconds.toFixed(1)}s`;
}
function setMode(value, reset=true) {
  mode=value;
  for (const m of ['mock','online']) { $(`${m}-mode`).classList.toggle('selected',m===mode); $(`${m}-mode`).setAttribute('aria-pressed',m===mode); }
  $('speed-field').hidden=mode!=='mock'; $('target-field').hidden=mode!=='online';
  $('mode-description').textContent=mode==='mock'?'回放真实的历史实验，每 5 轮展开一段新的发现。':'真实调用 DeepSeek Flash，使用 API 额度。每完成 5 轮更新一次实测结果。';
  $('start').innerHTML=mode==='mock'?'<span class="play-icon">▶</span> 开始回放':'<span class="play-icon">▶</span> 启动真实搜索';
  $('mode-badge').textContent=mode==='mock'?'HISTORICAL REPLAY':'ONLINE RESEARCH';
  $('live-label').textContent=mode==='mock'?'REPLAY':'LIVE';
  if (reset) { current=null; localStorage.removeItem('alphaldm-current'); renderKey=''; preview(); renderSessions(); }
}
function preview() {
  const item=archives.find(a=>a.seed===Number($('seed').value));
  const initial=mode==='mock'&&item?item.initial:{};
  render({...initial, seed:Number($('seed').value), mode, status:'ready', stage:'ready',
    committed_round:0,target_rounds:mode==='mock'?100:Number($('target').value),
    message:mode==='mock'?'初始因子库已就绪，等待播放':'启动后将进行真实的 LDM 搜索'});
}
function pointPath(points,x,y,step=false) {
  if(!points.length) return '';
  let p=`M${x(points[0].round)},${y(points[0].rank_ic)}`;
  for(const value of points.slice(1)) p+=step?`H${x(value.round)}V${y(value.rank_ic)}`:`L${x(value.round)},${y(value.rank_ic)}`;
  return p;
}
function drawChart(kind, points=[], envelope=[], target=100) {
  const el=$(kind+'-chart'), W=780,H=204,L=48,R=24,T=19,B=27;
  const valid=points.filter(p=>Number.isFinite(p.rank_ic));
  const color=kind==='train'?'#f0c27b':'#8ee0bb';
  const all=valid.map(p=>p.rank_ic);
  let lo=kind==='train'?.04:.02, hi=kind==='train'?.07:.05;
  if(all.length){lo=Math.min(lo,...all)-.001;hi=Math.max(hi,...all)+.001;}
  const x=r=>L+(W-L-R)*r/Math.max(5,target), y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  let svg=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${kind} RankIC from round 0 to ${target}">`;
  const tag=(text,xx,yy,cls='tick',anchor='end')=>`<text x="${xx}" y="${yy}" class="${cls}" text-anchor="${anchor}">${esc(text)}</text>`;
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4, yy=y(v);svg+=`<line class="grid" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/>`+tag(v.toFixed(3),L-9,yy+4);}
  for(let i=0;i<=5;i++){const r=target*i/5,xx=x(r);svg+=`<line class="grid-v" x1="${xx}" x2="${xx}" y1="${T}" y2="${H-B}"/>`+tag(Number.isInteger(r)?r:r.toFixed(1),xx,H-7,'tick','middle');}
  if(kind==='test'&&.035>=lo&&.035<=hi) svg+=`<line x1="${L}" x2="${W-R}" y1="${y(.035)}" y2="${y(.035)}" stroke="#657769" stroke-dasharray="4 5" stroke-width="1"/>`;
  // A failed measurement is a gap, never a zero or a new measured point.
  let segments=[],segment=[];
  for(const p of points){if(Number.isFinite(p.rank_ic))segment.push(p);else{segments.push(segment);segment=[];}}
  segments.push(segment);
  for(const group of segments){if(!group.length)continue;const path=pointPath(group,x,y,kind==='train');
    if(kind==='train'&&group.length>1)svg+=`<path d="${path}L${x(group.at(-1).round)},${H-B}L${x(group[0].round)},${H-B}Z" fill="#f0c27b" opacity=".045"/>`;
    svg+=`<path class="curve" d="${path}" stroke="${color}"/>`;
  }
  if(kind==='test'&&$('envelope').checked)svg+=`<path d="${pointPath(envelope,x,y,true)}" fill="none" stroke="#a9c8b6" stroke-width="1.6" opacity=".55" stroke-dasharray="4 4"/>`;
  valid.forEach((p,i)=>{svg+=`<circle class="point" data-index="${i}" cx="${x(p.round)}" cy="${y(p.rank_ic)}" r="${i===valid.length-1?4.3:2.8}" stroke="${color}" stroke-width="1.5" fill="#182124"/>`;});
  const last=valid.at(-1);
  if(last){const xx=x(last.round), yy=y(last.rank_ic),anchor=last.round>target*.85?'end':'start';svg+=`<circle cx="${xx}" cy="${yy}" r="8" fill="${color}" opacity=".09"/>`+`<text x="${xx+(anchor==='end'?-8:8)}" y="${Math.max(12,yy-12)}" fill="${color}" class="marker-label" text-anchor="${anchor}">R${last.round} · ${fmt(last.rank_ic)}</text>`;}
  else svg+=tag('等待第一个实测结果',W/2,H/2,'tick','middle');
  svg+='</svg>';el.innerHTML=svg;
  el.querySelectorAll('.point').forEach(circle=>{
    circle.addEventListener('pointerenter',event=>{const p=valid[Number(circle.dataset.index)],tip=$('tooltip');tip.textContent=`R${p.round} · ${fmt(p.rank_ic)}${p.factor?' · '+p.factor.name:' · Validation-selected Top-5'}`;tip.hidden=false;tip.style.left=Math.min(window.innerWidth-290,event.clientX+12)+'px';tip.style.top=Math.max(10,event.clientY-65)+'px';});
    circle.addEventListener('pointerleave',()=>$('tooltip').hidden=true);
  });
}
function render(value) {
  data=value;
  const round=value.committed_round||0,total=value.target_rounds||100,active=['queued','running'].includes(value.status);
  $('round').textContent=round; $('total').textContent=total;
  const percent=Math.min(100,Math.round(round/total*100));$('percent').textContent=percent+'%';$('progress').style.width=percent+'%';
  $('train-value').textContent=fmt(value.best_factor?.train_rank_ic);$('test-value').textContent=fmt(value.latest_test?.rank_ic);
  $('factor-count').textContent=value.factor_count??'—';$('observation-count').textContent=value.gp_observations?`${value.gp_observations} GP observations`:'按真实评估顺序保存';
  const baseline=value.train?.[0]?.rank_ic,train=value.best_factor?.train_rank_ic;
  $('train-delta').textContent=Number.isFinite(train)&&Number.isFinite(baseline)?`相对 R0 ${train-baseline>=0?'+':''}${(train-baseline).toFixed(4)}`:'已提交因子中的最佳表现';
  $('test-round').textContent=value.latest_test?`R${value.latest_test.round} · ${value.latest_test.qualified?'达到 0.035 阈值':'低于 0.035 阈值'}`:'等待阶段 Test 评估';
  $('run-status').textContent=statusNames[value.status]||'准备就绪';$('run-message').textContent=value.pause_requested&&active?'暂停请求已收到，等待安全保存':value.message||'';
  $('run-dot').className='status-dot '+(active?'running':value.status==='failed'?'failed':'ready');
  $('run-seed').textContent=`SEED ${value.seed??$('seed').value}`;
  $('run-time').textContent=value.mode==='mock'?`${value.interval_seconds||$('speed').value}s / 5 rounds`:`R${value.active_round??round} · 每 5 轮更新`;
  $('phase-name').textContent=phaseNames[value.stage]||'等待开始';$('phase-detail').textContent=value.mode==='mock'?'历史结果按原始顺序展示，不调用 LLM。':`${value.message||'实时搜索准备就绪。'}${value.phase_total?` · ${value.phase_completed||0}/${value.phase_total}`:''}`;
  $('round-duration').textContent=value.mode==='mock'?'历史回放':duration(value.last_round_seconds);
  $('stage-duration').textContent=value.mode==='online'&&active&&value.stage_started_at?duration(Date.now()/1000-value.stage_started_at):'—';
  document.querySelectorAll('.phase-steps span').forEach(el=>el.classList.toggle('active',value.mode==='online'&&(el.dataset.stage===value.stage||(el.dataset.stage==='surrogate'&&value.stage==='acquisition'))));
  const factor=value.best_factor;
  $('best-name').innerHTML=esc(factor?.name||'等待因子').replace(/_/g,'_<wbr>');$('best-expression').textContent=factor?.expression||'—';$('best-reason').textContent=factor?.reason||'真实评估后的最佳因子会展示在这里。';
  $('best-origin').textContent=factor?`TRAIN LEADER · ${fmt(factor.train_rank_ic)}`:'TRAIN LEADER';$('best-round').textContent=factor?`发现于 R${factor.round} · #${factor.evaluation_index}`:'—';$('copy-formula').disabled=!factor;
  $('recent').innerHTML=(value.recent_factors||[]).map(f=>`<div class="discovery"><span class="r">R${f.round}</span><span class="name" title="${esc(f.name)}">${esc(f.name)}</span><span class="score">${fmt(f.train_rank_ic)}</span></div>`).join('')||'<p class="muted">保留每个因子的生成顺序。</p>';
  $('top5-round').textContent=value.latest_test?`R${value.latest_test.round} · Validation Top-5`:'Validation Top-5';
  $('top5').innerHTML=(value.top5||[]).map((f,i)=>`<tr><td>${i+1}</td><td title="${esc(f.expression)}">${esc(f.name)}</td><td>R${f.round}</td><td>${fmt(f.validation_rank_ic)}</td></tr>`).join('')||'<tr><td colspan="4" class="empty">阶段评估完成后，在这里查看入选因子</td></tr>';
  $('train-checkpoint').textContent=value.train?.length?`R${value.train.at(-1).round} · 每 5 轮记录`:'等待初始因子库';
  $('pause').disabled=!current||!active||Boolean(value.pause_requested);
  $('speed').disabled=Boolean(current)&&active&&value.mode==='mock';
  $('resume').disabled=!current||!['paused','failed','interrupted','completed'].includes(value.status)||(value.status==='completed'&&value.mode==='mock')||value.alive===true;
  $('resume').textContent=value.status==='completed'?'续跑':'继续';$('export').disabled=!current;
  $('footer-state').textContent=current?`SESSION ${current.slice(0,8)} · ${value.mode==='mock'?'HISTORICAL REPLAY':'LIVE RESEARCH'}`:'Measured research. Visible progress.';
  const key=JSON.stringify([value.train,value.test,value.test_best_so_far,total,$('envelope').checked]);
  if(key!==renderKey){renderKey=key;drawChart('train',value.train||[],[],total);drawChart('test',value.test||[],value.test_best_so_far||[],total);}
  if(value.error)alertMessage(value.error);else if(value.last_provider_error)alertMessage(`模型接口响应异常：${value.last_provider_error}`);else if(value.reporting_warning)alertMessage(`搜索进度已保存，阶段报告待补齐：${value.reporting_warning}`);else if(pollErrors===0)alertMessage('');
  const item=jobs.find(j=>j.id===current);
  if(item&&(item.status!==value.status||item.committed_round!==round)){Object.assign(item,{status:value.status,committed_round:round});renderSessions();}
}
function renderSessions() {
  $('sessions').innerHTML=jobs.slice(0,8).map(j=>`<button class="session-item ${j.id===current?'active':''}" data-id="${j.id}"><span class="session-round">R${j.committed_round||0}</span><strong>${j.mode==='mock'?'Replay':'Online'} · Seed ${j.seed}</strong><small>${esc(statusNames[j.status]||j.status)} · ${new Date(j.created_at*1000).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}</small></button>`).join('')||'<p class="muted">还没有运行记录</p>';
  $('sessions').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>selectJob(button.dataset.id)));
}
async function sessions() {if(sessionBusy)return;sessionBusy=true;try{jobs=(await api('/api/runs')).runs;renderSessions();}catch{}finally{sessionBusy=false;}}
async function selectJob(id) {
  current=id;localStorage.setItem('alphaldm-current',id);renderKey='';
  try{const value=await api('/api/runs/'+id);if(current!==id)return;setMode(value.mode,false);$('seed').value=String(value.seed);$('target').value=value.target_rounds;$('speed').value=value.interval_seconds;$('speed-value').textContent=value.interval_seconds+' s';render(value);renderSessions();}
  catch(error){alertMessage(error.message);}
}
async function poll(){if(!current||pollBusy)return;pollBusy=true;const id=current;try{const value=await api('/api/runs/'+id);if(current===id){pollErrors=0;render(value);}}catch(error){pollErrors++;if(pollErrors>2)alertMessage('暂时无法连接服务，正在重连。后台任务与已保存进度会保留。');}finally{pollBusy=false;}}
async function start(){if(busy)return;busy=true;$('start').disabled=true;alertMessage('');try{
  const target=mode==='mock'?100:Number($('target').value);
  if(!Number.isInteger(target)||target<5||target>10000||target%5)throw new Error('目标轮数请输入 5 的倍数，范围 5–10000。');
  const job=await api('/api/runs',{method:'POST',body:JSON.stringify({mode,seed:Number($('seed').value),target_rounds:target,interval_seconds:Number($('speed').value)})});
  await selectJob(job.id);await sessions();
}catch(error){alertMessage(error.message);}finally{busy=false;$('start').disabled=false;}}
async function readiness(){try{const r=await api('/api/readiness');$('ready-dot').className='status-dot '+(r.online_ready?'ready':'');$('ready-text').textContent=r.online_ready?'Online 环境就绪':r.replay_seeds.length?'历史回放可用':'准备回放数据中';$('ready-text').title=`行情数据 ${r.market_data_available?'✓':'×'} · 评估服务 ${r.evaluator_ready?'✓':'×'} · API 凭据 ${r.credential_available?'✓':'×'}`;}catch{$('ready-text').textContent='服务连接中断';}}
$('mock-mode').onclick=()=>setMode('mock');$('online-mode').onclick=()=>setMode('online');$('seed').onchange=()=>{if(!current)preview();};
$('speed').oninput=()=>$('speed-value').textContent=$('speed').value+' s';$('target').onchange=()=>{if(!current)preview();};$('start').onclick=start;
$('pause').onclick=async()=>{if(!current)return;try{await api(`/api/runs/${current}/pause`,{method:'POST',body:'{}'});toast(mode==='mock'?'正在暂停回放':'将在本轮安全保存后暂停');await poll();}catch(error){alertMessage(error.message);}};
$('resume').onclick=async()=>{if(!current)return;try{const body={interval_seconds:Number($('speed').value)};if(mode==='online')body.target_rounds=Number($('target').value);await api(`/api/runs/${current}/resume`,{method:'POST',body:JSON.stringify(body)});await poll();await sessions();}catch(error){alertMessage(error.message);}};
$('envelope').onchange=()=>{if(data)render(data);};$('copy-formula').onclick=async()=>{try{await navigator.clipboard.writeText(data.best_factor.expression);toast('公式已复制');}catch{toast('请直接选择公式并复制');}};
$('export').onclick=()=>{if(current)window.location.href=`/api/runs/${current}/export`;};
async function init(){try{archives=(await api('/api/archives')).archives;await sessions();const saved=localStorage.getItem('alphaldm-current');if(saved&&jobs.some(j=>j.id===saved))await selectJob(saved);else preview();}catch(error){alertMessage(error.message);preview();}readiness();setInterval(poll,1000);setInterval(sessions,4000);setInterval(readiness,30000);}
init();
