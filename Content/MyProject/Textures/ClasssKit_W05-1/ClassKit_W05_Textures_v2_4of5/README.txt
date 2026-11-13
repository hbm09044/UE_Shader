ClassKit_W05 텍스처 v2 (2026-11-12)
====================================
기존 5주차 텍스처를 같은 파일명으로 다시 만든 것입니다. 머티리얼 그래프와 인스턴스는 그대로 씁니다.

파일 (각 2048 x 2048 PNG)
  T_Kit_Dirt_BC / _N / _ORM
  T_Kit_Gravel_BC / _N / _ORM
  T_Kit_Grass_BC / _N / _ORM
  T_Kit_Snow_BC / _N
  _BC  : RGB = 색(sRGB), A = Height
  _N   : DirectX 노멀
  _ORM : R = AO, G = Roughness, B = Metallic(전부 0)

덮어쓰기 임포트
  1. zip 5개를 한 폴더에 풉니다.
  2. PNG 11장을 Content Drawer의 기존 텍스처가 있는 폴더로 드래그합니다.
  3. 같은 이름이 있다는 창이 뜨면 전부 덮어쓰기를 고릅니다.
  4. 임포트 뒤 확인: _ORM 3장은 sRGB 해제 + Compression = Masks (no sRGB),
     _BC는 sRGB 켬, _N은 Normalmap. 달라졌으면 고치고 Save.

Tools
  texlib.py / make_w05_textures.py : 생성 스크립트 (python make_w05_textures.py <출력폴더>)
  verify_w05.py : 이음매·하이트 분포 검사, Lerp vs HeightLerp 비교 렌더
Preview
  compare_Lerp_vs_HeightLerp.png : 같은 마스크에서 왼쪽 Lerp, 오른쪽 HeightLerp
