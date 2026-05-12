# 복지 데이터 스키마 문서

임산부/영유아 가정을 위한 맞춤형 복지 혜택 RAG 챗봇에 사용되는 데이터 구조를 설명합니다.

---

## 파일 구조

```
data/
├── welfare_data.json   # 복지 정책 데이터 
└── README.md           # 본 문서
```

---

## JSON 스키마 설명

### 기본 정보

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `id` | string | 정책 고유 ID. `{지역}_{카테고리}_{번호}` 형식 | `"seoul_traffic_001"` |
| `policy_name` | string | 정책 이름 | `"임산부 교통비 지원"` |
| `category` | string | 정책 분류 | `"교통비"`, `"의료비"`, `"현금"`, `"바우처"` |
| `district` | string | 정책이 **적용되는 지역** | `"서울"`, `"전국"`, `"경기 수원시"` |
| `agency` | string | 정책을 **운영하는 기관** | `"서울특별시"`, `"보건복지부"` |

> `district`와 `agency`를 구분하는 이유: 기관과 적용 지역이 다를 수 있음.  
> 예) 강남구청(agency)이 운영하지만 서울 강남구(district) 주민만 신청 가능한 경우.

---

### 정책 내용

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `content` | string | **RAG 임베딩 대상**. 정책의 핵심 내용을 자연어로 요약 (300~800자 권장) | `"임산부 1인당 교통비 바우처 70~100만원 지급..."` |
| `target` | string | 신청 대상 자격 조건 설명 | `"신청일 기준 서울시 거주 임산부..."` |
| `benefit_type` | string | 혜택 종류 | `"바우처"`, `"현금"`, `"서비스"` |
| `benefit_amount` | string | 혜택 금액 또는 규모 | `"70~100만원"` |

---

### 신청 정보

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `apply_method` | string | 신청 방법 | `"온라인(복지로) 또는 방문(주민센터)"` |
| `apply_period` | string | 신청 가능 기간 | `"임신 12주차 ~ 출산 후 6개월"` |
| `use_period` | string | 혜택 사용 가능 기간 | `"분만예정일로부터 12개월"` |
| `inquiry` | string | 문의처 | `"다산콜센터 120"` |
| `url` | string | 정책 상세 페이지 URL | `"https://..."` |

---

### 필터링 조건 (RAG 메타데이터 필터용)

| 필드 | 타입 | 설명 | 값 규칙 |
|------|------|------|---------|
| `pregnancy_status` | array | 신청 가능한 상태 | `"임신중"`, `"출산후"` 중 해당하는 것 |
| `pregnancy_weeks_min` | int \| null | 신청 가능한 최소 임신 주차 | `12` = 12주차 이상, `null` = 제한 없음 |
| `infant_months_max` | int \| null | 신청 가능한 자녀 최대 월령 | `6` = 생후 6개월 이하, `null` = 제한 없음 |
| `user_age_max` | int \| null | 신청 가능한 최대 연령 (청소년산모 등) | `19` = 만 19세 이하, `null` = 연령 제한 없음 |
| `income_level_max` | int \| null | 소득 기준 (중위소득 기준 %) | `10` = 소득 무관 (전체 허용), `1~9` = 중위소득 10%~90% |
| `is_multibirth` | bool \| null | 다태아(쌍둥이 등) 전용 여부 | `true` = 다태아 전용, `false` = 단태아, `null` = 무관 |
| `is_foreigner` | bool | 외국인 신청 가능 여부 | `true` = 가능, `false` = 불가 |
| `foreigner_condition` | string \| null | 외국인 신청 조건 상세 | `is_foreigner`가 `true`일 때만 작성 |

---

### 태그 및 관리

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `tags` | array | 키워드 검색 보조용 태그 | `["교통비", "바우처", "서울"]` |
| `is_active` | bool | 현재 운영 중인 정책 여부 | `true` = 운영 중, `false` = 종료 |
| `updated_at` | string | 마지막 데이터 업데이트 날짜 | `"2025-01"` |

---

## ID 작성 규칙

```
{지역코드}_{카테고리코드}_{번호}

지역코드: seoul / gyeonggi / busan / national (전국) 등
카테고리코드: traffic / medical / cash / voucher / service 등
번호: 001부터 순서대로

예시:
- seoul_traffic_001    → 서울 교통비 지원 1번
- national_voucher_001 → 전국 바우처 지원 1번
- gyeonggi_cash_002    → 경기도 현금 지원 2번
```
