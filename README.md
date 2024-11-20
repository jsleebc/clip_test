# NFT 이미지 처리 프로젝트

## 프로젝트 구조

이 프로젝트는 NFT 이미지의 학습 및 테스트를 위한 다양한 파일과 폴더로 구성되어 있습니다.

### 파일 및 폴더 설명

- **test.py**: 학습을 실행하는 파일입니다.
- **model_results/**: 모델을 저장하기 위해 생성된 폴더입니다.
- **train.py**: 학습된 모델을 불러와서 테스트하는 파일입니다.
- **train_image/**: 학습에 사용되는 이미지와 메타데이터가 포함된 폴더입니다.
- **metadata/**: Bored Ape Yacht Club (BAYC) 관련 8000개 이미지의 메타데이터가 포함된 폴더입니다.
- **test_image/**: 테스트에 사용되는 BAYC 관련 2000개 이미지가 포함된 폴더입니다.
- **description.py**: 메타데이터를 학습에 사용되는 텍스트로 변환하는 코드가 포함된 파일입니다.

### clip_descriptions.json

각 파일에 대한 학습 예시를 포함하고 있습니다:
```json
{
  "bayc_0201": "The Bored Ape Yacht Club is a unique digital collectible NFT. This ape has new punk blue background, black fur, bored eyes, bored unshaven mouth, wearing leather punk jacket.",
  "bayc_6207": "The Bored Ape Yacht Club is a unique digital collectible NFT. This ape has new punk blue background, noise fur, blue beams eyes, bored bubblegum mouth, wearing puffy vest, wearing bunny ears hat.",
  "bayc_2011": "The Bored Ape Yacht Club is a unique digital collectible NFT. This ape has new punk blue background, white fur, sleepy eyes, bored unshaven mouth, wearing guayabera."
}