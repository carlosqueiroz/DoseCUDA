 python - <<'PY'
> import numpy as np, pydicom
> np.set_printoptions(precision=4)
> path_npy='/home/rt/scripts/DoseCUDA/tests/test_patient_output/dose_calculated.npy'
> arr=np.load(path_npy)
> print('npy:', path_npy)
> print('shape:', arr.shape)
> print('dtype:', arr.dtype)
> print('min/max/mean:', float(arr.min()), float(arr.max()), float(arr.mean()))
> 
> # Inspect RTDOSE
> from pydicom import dcmread
> path_dcm='/home/rt/scripts/DoseCUDA/tests/test_patient_output/DoseCUDA_RD.dcm'
> d=dcmread(path_dcm, stop_before_pixels=True)
> print('\nRTDOSE metadata:')
> for tag in ['Rows','Columns','NumberOfFrames','BitsAllocated','DoseUnits','DoseType','DoseSummationType','Manufacturer']:
>     if hasattr(d,tag):
>         print(f'{tag}:', getattr(d,tag))
>     else:
>         print(f'{tag}: <missing>')
> 
> # Check Pixel Data presence
> try:
>     ds=dcmread(path_dcm)
>     print('\nPixel data length:', len(ds.PixelData))
> except Exception as e:
>     print('Error reading PixelData:', e)
> PY
npy: /home/rt/scripts/DoseCUDA/tests/test_patient_output/dose_calculated.npy
shape: (134, 201, 201)
dtype: float32
min/max/mean: 0.0 0.13731764256954193 3.0498953492497094e-05

RTDOSE metadata:
Rows: 260
Columns: 472
NumberOfFrames: 341
BitsAllocated: 32
DoseUnits: GY
DoseType: PHYSICAL
DoseSummationType: PLAN
Manufacturer: Varian Medical Systems

Pixel data length: 167390080
(.venv) rt@dev1:~/scripts/DoseCUDA$  python - <<'PY'
> from pydicom import dcmread
> d=dcmread('/home/rt/scripts/DoseCUDA/tests/test_patient_output/DoseCUDA_RD.dcm')
> keys=['DoseGridScaling','RescaleIntercept','RescaleSlope','PixelRepresentation','HighBit','SmallestImagePixelValue','LargestImagePixelValue']
> print('Checking DICOM dose scaling tags:')
> for k in keys:
>     print(k, getattr(d,k, '<missing>'))
> # Print first few pixel values after applying scaling if possible
> import numpy as np
> arr = d.pixel_array.astype('float32')
> print('\nPixel array shape:', arr.shape)
> print('Sample pixel values (frame 0) min/max/mean:', float(arr[0].min()), float(arr[0].max()), float(arr[0].mean()))
> # If DoseGridScaling present, compute physical dose
> if hasattr(d,'DoseGridScaling'):
>     scale=d.DoseGridScaling
>     print('DoseGridScaling:', scale)
>     phys=arr*scale
>     print('physical dose sample min/max/mean:', float(phys.min()), float(phys.max()), float(phys.mean()))
> PY
Checking DICOM dose scaling tags:
DoseGridScaling 1e-10
RescaleIntercept <missing>
RescaleSlope <missing>
PixelRepresentation 0
HighBit 31
SmallestImagePixelValue <missing>
LargestImagePixelValue <missing>

Pixel array shape: (341, 260, 472)
Sample pixel values (frame 0) min/max/mean: 0.0 0.0 0.0
DoseGridScaling: 1e-10
physical dose sample min/max/mean: 0.0 0.1324070543050766 4.758737122756429e-05
(.venv) rt@dev1:~/scripts/DoseCUDA$  python - <<'PY'
> from pydicom import dcmread
> d=dcmread('/home/rt/scripts/DoseCUDA/tests/test_patient_output/RTDOSE_template.dcm', stop_before_pixels=True)
> print('Template DoseGridScaling:', getattr(d,'DoseGridScaling','<missing>'))
> print('BitsAllocated:', getattr(d,'BitsAllocated','<missing>'))
> print('Rows,Columns,NumberOfFrames:', getattr(d,'Rows','?'), getattr(d,'Columns','?'), getattr(d,'NumberOfFrames','?'))
> PY
Template DoseGridScaling: 2.9670870618e-05
BitsAllocated: 32
Rows,Columns,NumberOfFrames: 260 472 341
(.venv) rt@dev1:~/scripts/DoseCUDA$  python - <<'PY'
> # A: Debug dose max by loading saved numpy dose and showing stats
> import numpy as np
> path='/home/rt/scripts/DoseCUDA/tests/test_patient_output/dose_calculated.npy'
> arr=np.load(path)
> print('A1: Loaded dose numpy')
> print('shape',arr.shape,'dtype',arr.dtype)
> print('min/max/mean',float(arr.min()),float(arr.max()),float(arr.mean()))
> # Also print where max occurs
> idx=arr.argmax()
> z,y,x = np.unravel_index(idx, arr.shape)
> print('max at (z,y,x)=', (int(z),int(y),int(x)), 'value=', float(arr[z,y,x]))
> # Print corresponding NRRD header scaling (none) and RTDOSE scaling
> from pydicom import dcmread
> d=dcmread('/home/rt/scripts/DoseCUDA/tests/test_patient_output/DoseCUDA_RD.dcm', stop_before_pixels=True)
> print('A2: RTDOSE template-derived DoseGridScaling in file:', getattr(d,'DoseGridScaling',None))
> PY
A1: Loaded dose numpy
shape (134, 201, 201) dtype float32
min/max/mean 0.0 0.13731764256954193 3.0498953492497094e-05
max at (z,y,x)= (59, 73, 0) value= 0.13731764256954193
A2: RTDOSE template-derived DoseGridScaling in file: 1e-10
(.venv) rt@dev1:~/scripts/DoseCUDA$  python - <<'PY'
> # B: Try to locate Plan object saved during tests; tests may save plan to test_patient_output/phase_*/
> import json, os
> outdir='/home/rt/scripts/DoseCUDA/tests/test_patient_output'
> # Search for possible pickled plan files
> for root,dirs,files in os.walk(outdir):
>     for f in files:
>         if f.endswith('.pkl') or f.endswith('.plan') or f.endswith('.npz'):
>             print('Found candidate:', os.path.join(root,f))
> # If not found, inspect phase directories for plan metadata
> for phase in ['phase_1','phase_2']:
>     pd=os.path.join(outdir,phase)
>     if os.path.exists(pd):
>         print('\nPhase dir:',pd)
>         print('files:', os.listdir(pd))
>         # look for json
>         for f in os.listdir(pd):
>             if f.endswith('.json') or f.endswith('.txt'):
>                 print(' -',f)
> # Fallback: attempt to recreate Plan via parsing RTPLAN
> from DoseCUDA import dicom_case_discovery, plan, plan_imrt
> # Try to enumerate phases
> cases = list(dicom_case_discovery.enumerate_phases('/home/rt/scripts/DoseCUDA/tests/test_patient_output'))
> print('\nEnumerated phases count:', len(cases))
> for c in cases:
>     print('Phase:', c.name, 'rtplan:', getattr(c,'rtplan',None))
> PY

Phase dir: /home/rt/scripts/DoseCUDA/tests/test_patient_output/phase_1
files: ['dose.npy', 'CT']

Phase dir: /home/rt/scripts/DoseCUDA/tests/test_patient_output/phase_2
files: ['dose.npy', 'CT']
/home/rt/scripts/DoseCUDA/DoseCUDA/plan_imrt.py:9: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
Traceback (most recent call last):
  File "<stdin>", line 22, in <module>
  File "/home/rt/scripts/DoseCUDA/DoseCUDA/dicom_case_discovery.py", line 679, in enumerate_phases
    for plan in case.rtplan_files:
                ^^^^^^^^^^^^^^^^^
AttributeError: 'str' object has no attribute 'rtplan_files'
(.venv) rt@dev1:~/scripts/DoseCUDA$  python - <<'PY'
> # B continuation: load phase dose.npy files to inspect per-phase dose
> import numpy as np, os
> for phase in ['phase_1','phase_2']:
>     p=os.path.join('/home/rt/scripts/DoseCUDA/tests/test_patient_output',phase,'dose.npy')
>     if os.path.exists(p):
>         a=np.load(p)
>         print(phase,'shape',a.shape,'min/max/mean',float(a.min()),float(a.max()),float(a.mean()))
>     else:
>         print(phase,'dose.npy missing')
> # Also attempt to construct PlanImrt object from RTPLAN to inspect beams
> from DoseCUDA.plan_imrt import PlanIMRT
> rtplan_path='/home/rt/scripts/DoseCUDA/tests/test_patient_output/RTPLAN.dcm'
> try:
>     p=PlanIMRT(rtplan_path)
>     print('\nPlanIMRT beams count:', len(p.beams))
>     for i,b in enumerate(p.beams[:10]):
>         print('Beam',i+1, 'id', getattr(b,'beam_id',None), 'MU', getattr(b,'MU',None))
> except Exception as e:
>     print('PlanIMRT load error:',e)
> PY
phase_1 shape (134, 201, 201) min/max/mean 0.0 0.005738086067140102 2.2146827177493833e-05
phase_2 shape (134, 201, 201) min/max/mean 0.0 0.13731764256954193 3.0498953492497094e-05
/home/rt/scripts/DoseCUDA/DoseCUDA/plan_imrt.py:9: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
Traceback (most recent call last):
  File "<stdin>", line 11, in <module>
ImportError: cannot import name 'PlanIMRT' from 'DoseCUDA.plan_imrt' (/home/rt/scripts/DoseCUDA/DoseCUDA/plan_imrt.py)
(.venv) rt@dev1:~/scripts/DoseCUDA$  python - <<'PY'
> # B alternative: inspect RTPLAN.dcm for beam control points / MUs via pydicom
> from pydicom import dcmread
> rtplan='/home/rt/scripts/DoseCUDA/tests/test_patient_output/RTPLAN.dcm'
> d=dcmread(rtplan)
> print('RTPLAN SOPClassUID', d.SOPClassUID)
> # Try to find BeamSequence
> if hasattr(d,'BeamSequence'):
>     print('BeamSequence length:', len(d.BeamSequence))
>     for i,beam in enumerate(d.BeamSequence):
>         print('\nBeam',i+1)
>         print('BeamNumber:', getattr(beam,'BeamNumber',None))
>         print('NominalBeamEnergy:', getattr(beam,'NominalBeamEnergy',None))
>         print('DeliveryControlPointSequence length:', len(getattr(beam,'ControlPointSequence',[])))
>         # MU info may be in FractionGroupSequence
> else:
>     print('No BeamSequence found; checking FractionGroupSequence')
> if hasattr(d,'FractionGroupSequence'):
>     print('FractionGroupSequence length', len(d.FractionGroupSequence))
>     for fg in d.FractionGroupSequence:
>         print('ReferencedBeamSequence length:', len(getattr(fg,'ReferencedBeamSequence',[])))
>         for rb in getattr(fg,'ReferencedBeamSequence',[]):
>             print(' Ref BeamNumber', getattr(rb,'ReferencedBeamNumber',None))
> # Try to find beam MUs in BeamSequence->ControlPointSequence->CumulativeMetersetWeight
> for i,beam in enumerate(getattr(d,'BeamSequence',[])):
>     for cp in getattr(beam,'ControlPointSequence',[]):
>         if hasattr(cp,'CumulativeMetersetWeight'):
>             print('Beam',i+1,'has CumulativeMetersetWeight in first CP:', cp.CumulativeMetersetWeight)
>             break
> PY
RTPLAN SOPClassUID 1.2.840.10008.5.1.4.1.1.481.5
BeamSequence length: 7

Beam 1
BeamNumber: 1
NominalBeamEnergy: None
DeliveryControlPointSequence length: 2

Beam 2
BeamNumber: 3
NominalBeamEnergy: None
DeliveryControlPointSequence length: 2

Beam 3
BeamNumber: 4
NominalBeamEnergy: None
DeliveryControlPointSequence length: 2

Beam 4
BeamNumber: 7
NominalBeamEnergy: None
DeliveryControlPointSequence length: 2

Beam 5
BeamNumber: 2
NominalBeamEnergy: None
DeliveryControlPointSequence length: 2

Beam 6
BeamNumber: 5
NominalBeamEnergy: None
DeliveryControlPointSequence length: 2

Beam 7
BeamNumber: 6
NominalBeamEnergy: None
DeliveryControlPointSequence length: 2
FractionGroupSequence length 1
ReferencedBeamSequence length: 7
 Ref BeamNumber 1
 Ref BeamNumber 3
 Ref BeamNumber 4
 Ref BeamNumber 7
 Ref BeamNumber 2
 Ref BeamNumber 5
 Ref BeamNumber 6
Beam 1 has CumulativeMetersetWeight in first CP: 0
Beam 2 has CumulativeMetersetWeight in first CP: 0
Beam 3 has CumulativeMetersetWeight in first CP: 0
Beam 4 has CumulativeMetersetWeight in first CP: 0
Beam 5 has CumulativeMetersetWeight in first CP: 0
Beam 6 has CumulativeMetersetWeight in first CP: 0
Beam 7 has CumulativeMetersetWeight in first CP: 0
(.venv) rt@dev1:~/scripts/DoseCUDA$  python - <<'PY'
> from pydicom import dcmread
> import numpy as np
> rt_dcm='/home/rt/scripts/DoseCUDA/tests/test_patient_output/DoseCUDA_RD.dcm'
> template='/home/rt/scripts/DoseCUDA/tests/test_patient_output/RTDOSE_template.dcm'
> rd=dcmread(rt_dcm)
> px=rd.pixel_array.astype('float32')
> print('pixel array shape', px.shape)
> tpl=dcmread(template, stop_before_pixels=True)
> template_scale=getattr(tpl,'DoseGridScaling',None)
> current_scale=getattr(rd,'DoseGridScaling',None)
> print('current_scale',current_scale,'template_scale',template_scale)
> phys_current=px*current_scale
> phys_template=px*template_scale
> print('current phys max/min/mean:', float(phys_current.max()), float(phys_current.min()), float(phys_current.mean()))
> print('template phys max/min/mean:', float(phys_template.max()), float(phys_template.min()), float(phys_template.mean()))
> # Show ratio
> print('max ratio template/current:', float(phys_template.max()/max(phys_current.max(),1e-20)))
> PY
pixel array shape (341, 260, 472)
current_scale 1e-10 template_scale 2.9670870618e-05
current phys max/min/mean: 0.1324070543050766 0.0 4.758737122756429e-05
template phys max/min/mean: 39286.328125 0.0 14.119586944580078
max ratio template/current: 296708.71875
(.venv) rt@dev1:~/scripts/DoseCUDA$  python - <<'PY'
> from pydicom import dcmread
> rtplan='/home/rt/scripts/DoseCUDA/tests/test_patient_output/RTPLAN.dcm'
> ds=dcmread(rtplan)
> fg=ds.FractionGroupSequence[0]
> print('NumberOfFractionsPlanned', getattr(fg,'NumberOfFractionsPlanned',None))
> for ref in fg.ReferencedBeamSequence:
>     print('ReferencedBeamNumber', ref.ReferencedBeamNumber, 'BeamMeterset', getattr(ref,'BeamMeterset',None))
> # Also show BeamSequence BeamMeterset if present
> for i,beam in enumerate(ds.BeamSequence):
>     print('Beam seq',i+1,'BeamNumber', getattr(beam,'BeamNumber',None), 'FinalCumulativeMetersetWeight', getattr(beam,'FinalCumulativeMetersetWeight',None))
> PY
NumberOfFractionsPlanned 26
ReferencedBeamNumber 1 BeamMeterset 127.306097094592
ReferencedBeamNumber 3 BeamMeterset 10.0658073046762
ReferencedBeamNumber 4 BeamMeterset 9.92316510999585
ReferencedBeamNumber 7 BeamMeterset 23.0789347637067
ReferencedBeamNumber 2 BeamMeterset 127.642616006713
ReferencedBeamNumber 5 BeamMeterset 9.64187125759027
ReferencedBeamNumber 6 BeamMeterset 10.0073160024865
Beam seq 1 BeamNumber 1 FinalCumulativeMetersetWeight 1
Beam seq 2 BeamNumber 3 FinalCumulativeMetersetWeight 1
Beam seq 3 BeamNumber 4 FinalCumulativeMetersetWeight 1
Beam seq 4 BeamNumber 7 FinalCumulativeMetersetWeight 1
Beam seq 5 BeamNumber 2 FinalCumulativeMetersetWeight 1
Beam seq 6 BeamNumber 5 FinalCumulativeMetersetWeight 1
Beam seq 7 BeamNumber 6 FinalCumulativeMetersetWeight 1
(.venv) rt@dev1:~/scripts/DoseCUDA$ 

Vou aplicar uma correção em plan_imrt.py para gerar CumulativeMetersetWeight sintético quando ausente, de forma a não produzir segmentos com MU zero. Em seguida rodarei um script rápido para verificar doses por fase.

 esta certo isso? por que o CumulativeMetersetWeight esta zero? falha no modelo? pois é caso clinico  o dicom... ..
 
 
 
 Concluí A, B e C e summarize abaixo:

O que fiz

A: Carreguei dose_calculated.npy — shape (134,201,201), max=0.1373 Gy (localizado em z,y,x=(59,73,0)).
B: Inspecionei RTPLAN.dcm — FractionGroupSequence contém ReferencedBeamSequence com BeamMeterset para cada beam (valores ≈ 9–127). BeamSequence tem FinalCumulativeMetersetWeight=1 e os CumulativeMetersetWeight nos CPs são 0/1 (portanto segment MUs calculados podem ser muito pequenos/zero).
C: Recalculei estatísticas físicas do RTDOSE usando:
escala atual no arquivo DoseCUDA_RD.dcm: DoseGridScaling = 1e-10 → phys max ≈ 0.1324 Gy (consistente com NPY)
escala no template RTDOSE_template.dcm: DoseGridScaling ≈ 2.967e-05 → aplicá-la aos mesmos pixels daria ~39286 Gy (incompatível — porque os pixels foram quantizados com escala diferente).
Diagnóstico rápido

Os arquivos de dose são gerados corretamente (NPY, NRRD, DICOM presentes), porém os valores calculados são muito baixos (0.1 Gy) vs referência (65–72 Gy).
BeamMeterset existe, mas o parsing das CumulativeMetersetWeight (0→1) significa que segment MUs podem estar sendo calculados como zero para muitos segmentos (o código ignora segmentos com seg_mu <= 1e-9), resultando em quase nenhum dose entregue.
Avisos anteriores (kernel_weights sum = 3.000 e MU sanity) indicam problemas de normalização do modelo de feixe ou do próprio RTPLAN — essas inconsistências explicam a dose baixa.
Recomendações imediatas (posso executar)

Instrumentar plan_imrt.py no ponto onde é calculado seg_mu para imprimir beam_number, delta_cmw, scaling_factor, seg_mu e contagem de segmentos ignorados; reexecutar pipeline para confirmar quantos segmentos têm MU≈0. (Recomendado)
Verificar carregamento dos modelos de feixe (beam_models) e normalização das kernel_weights — corrigir soma ≠1 se necessário.
Se preferir validação rápida, posso alterar temporariamente o critério de ignorar seg_mu <= 1e-9 para um limiar menor e reexecutar (apenas para debug).