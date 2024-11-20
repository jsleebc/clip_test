# NFT 이미지 처리 프로젝트

## 프로젝트 개요
이 프로젝트는 NFT 이미지의 학습 및 테스트를 위한 시스템을 구현합니다. Bored Ape Yacht Club (BAYC) NFT 컬렉션을 대상으로 이미지 유사도 분석과 특성 인식을 수행합니다.

## 프로젝트 구조

### 파일 및 폴더
- **test.py**: 모델 학습 실행 파일
- **train.py**: 학습된 모델을 사용한 테스트 실행 파일
- **description.py**: 메타데이터를 학습용 텍스트로 변환하는 파일
- **model_results/**: 학습된 모델 저장 폴더
- **train_image/**: 학습용 이미지 및 메타데이터 (8000개 BAYC 이미지)
- **test_image/**: 테스트용 이미지 (2000개 BAYC 이미지)
- **metadata/**: BAYC 메타데이터 폴더

## 데이터 형식

### clip_descriptions.json
```json
{
  "bayc_0201": "The Bored Ape Yacht Club is a unique digital collectible NFT. This ape has new punk blue background, black fur, bored eyes, bored unshaven mouth, wearing leather punk jacket.",
  "bayc_6207": "The Bored Ape Yacht Club is a unique digital collectible NFT. This ape has new punk blue background, noise fur, blue beams eyes, bored bubblegum mouth, wearing puffy vest, wearing bunny ears hat.",
  "bayc_2011": "The Bored Ape Yacht Club is a unique digital collectible NFT. This ape has new punk blue background, white fur, sleepy eyes, bored unshaven mouth, wearing guayabera."
}
```

## 실행 방법
1. 모델 학습:
```bash
python test.py
```

2. 테스트 실행:
```bash
python train.py
```

## 결과 형식

테스트 결과는 `similarity_test_results.json` 파일에 저장됩니다.

### 결과 예시
```json
{
  "bayc_3815.png": {
    "image_similarity": {
      "top_10": [
        {
          "image": "bayc_3367.png",
          "similarity": 0.9397972226142883
        },
        ...
      ],
      "bottom_10": [
        {
          "image": "bayc_1581.png",
          "similarity": 0.3021571934223175
        },
        ...
      ]
    },
    "text_similarity": {
      "general_recognition": {
        "this is nft artwork": 0.048681002110242844,
        "this is bored ape yacht club": 0.052362147718667984,
        ...
      },
      "trait_recognition": {
        "a bored ape with gold fur": 0.08711913228034973,
        "a bored ape with robot fur": 0.00038238632259890437,
        ...
      }
    }
  }
}
```

## 참고 논문

1. [On the Evolution of (Hateful) Memes by Means of Multimodal Contrastive Learning](https://arxiv.org/abs/2212.06573)
2. [Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models](https://arxiv.org/abs/2305.13873)

### 관련 코드
- [Unsafe Diffusion GitHub](https://github.com/YitingQu/unsafe-diffusion)
- [Meme Evolution GitHub](https://github.com/YitingQu/meme-evolution)