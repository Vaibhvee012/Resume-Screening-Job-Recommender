import sys, warnings
from streamlit.testing.v1 import AppTest
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent.parent))
at=AppTest.from_file(str(__import__('pathlib').Path(__file__).resolve().parent.parent/'app.py'),default_timeout=240).run()
def chk(label):
    print(label,'| exceptions:',[e.value for e in at.exception],'| errors:',[e.value[:90] for e in at.error],'| warnings:',[w.value[:70] for w in at.warning])
chk('dashboard-empty')
for p in ['Resume Analyzer','Job Matcher','Job Recommendations','Skill Gap','Model Evaluation','About Project']:
    at.session_state['nav']=p; at.run(); chk(p)
at.session_state['nav']='Resume Analyzer'; at.run()
at.selectbox[0].set_value('sample_data_scientist.txt').run(); chk('resume loaded')
print([m.value[:80] for m in at.success])
for p in ['Job Matcher','Job Recommendations','Skill Gap','Dashboard']:
    at.session_state['nav']=p; at.run(); chk(p+' w/ resume')

# ---- interactive flows
at.session_state['nav']='Job Matcher'; at.run()
[b for b in at.button if b.label=='Analyse & match'][0].click().run(); chk('matcher click')
print('match overall:', at.session_state['match']['overall'], '| job:', at.session_state['job']['title'])
[r for r in at.radio if r.label=='Job source'][0].set_value('Paste a job description').run()
[b for b in at.button if b.label=='Load sample job description'][0].click().run()
[b for b in at.button if b.label=='Analyse & match'][0].click().run(); chk('paste click')
print('pasted overall:', at.session_state['match']['overall'], at.session_state['job']['title'])
at.text_area[0].set_value('   ').run(); [b for b in at.button if b.label=='Analyse & match'][0].click().run(); chk('empty JD')
at.session_state['nav']='Job Recommendations'; at.run()
[b for b in at.button if b.label=='Analyse skill gap for this job'][0].click().run(); chk('rec->gap')
print('nav now:', at.session_state['nav'], '| job:', at.session_state['job']['title'])
at.session_state['nav']='Dashboard'; at.run(); chk('dashboard final')
print('dashboard charts:', len(at.get('plotly_chart')))
