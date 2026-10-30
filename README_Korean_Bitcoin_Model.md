# 비트코인 예측 모델 실행 안내서

**학생:** SOBITOV ABUBAKIR AZAMAT UGLI\
**과목:** 정보분석특론\
**프로젝트:** 비트코인 예측 모델 구축 및 예측 성능 분석

## 1. 프로젝트 소개

이 프로젝트는 Binance Public API에서 수집한 비트코인 시간별 시장
데이터를 사용하여 예측 모델을 학습하고 평가합니다. 또한 Streamlit
대시보드를 통해 BTC 가격, 변동성 예측 및 가격 방향 예측 결과를 확인할 수
있습니다.

## 2. 실행 환경

-   Python 3
-   macOS 또는 Python을 실행할 수 있는 환경
-   인터넷 연결 (시장 데이터 및 환율 정보를 불러올 때 필요)

## 3. 프로젝트 파일

  -----------------------------------------------------------------------
  파일/폴더                           설명
  ----------------------------------- -----------------------------------
  `download_data.py`                  Binance Public API에서 시장
                                      데이터를 수집합니다.

  `prepare_data.py`                   데이터를 정리하고 모델 입력 특성을
                                      준비합니다.

  `evaluate_volatility.py`            변동성 예측 모델을 평가합니다.

  `direction_model.py`                가격 상승/하락 방향 분류 모델을
                                      평가합니다.

  `predict_volatility.py`             저장된 모델로 변동성을 예측합니다.

  `app.py`                            Streamlit 대시보드를 실행합니다.

  `data/`                             수집 데이터와 분석 결과 파일을
                                      저장합니다.

  `models/`                           학습된 모델 파일을 저장합니다.
  -----------------------------------------------------------------------

※ 실제 프로젝트 폴더에 없는 파일은 실행 전에 확인해 주세요.

## 4. 설치 방법

터미널에서 프로젝트 폴더로 이동합니다.

``` bash
cd ~/Desktop/App/BitCoin_Model
```

가상환경을 만들고 활성화합니다.

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

필요한 라이브러리를 설치합니다.

``` bash
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn statsmodels joblib streamlit requests
```

## 5. 실행 방법

### A. 데이터 수집

``` bash
python download_data.py
```

이 명령은 Binance Public API에서 시장 데이터를 가져옵니다. 인터넷 연결이
필요합니다. 이미 데이터 파일이 있고 다시 다운로드할 필요가 없다면 이
단계는 건너뛸 수 있습니다.

### B. 데이터 전처리

``` bash
python prepare_data.py
```

수집한 데이터를 모델 학습과 평가에 사용할 수 있도록 준비합니다.

### C. 변동성 모델 평가

``` bash
python evaluate_volatility.py
```

Ridge Regression, Random Forest 및 기준 모델의 MAE/RMSE 결과를
확인합니다.

### D. 가격 방향 분류 평가

``` bash
python direction_model.py
```

Logistic Regression, Random Forest 및 기준 모델의 분류 성능을
확인합니다.

### E. 변동성 예측

``` bash
python predict_volatility.py
```

저장된 모델이 있으면 변동성 예측 결과를 출력합니다.

### F. 대시보드 실행

``` bash
streamlit run app.py
```

터미널에 표시된 로컬 주소를 브라우저에서 엽니다. 대시보드를 중지하려면
터미널에서 `Control + C`를 누릅니다.

## 6. 평가 지표

-   **MAE (Mean Absolute Error):** 예측값과 실제값 사이의 평균 절대
    오차입니다. 일반적으로 낮을수록 좋습니다.
-   **RMSE (Root Mean Squared Error):** 큰 오차에 더 큰 페널티를 주는
    오차 지표입니다. 일반적으로 낮을수록 좋습니다.
-   **Accuracy:** 분류 예측 중 정답의 비율입니다.
-   **Balanced Accuracy, Precision, Recall, F1:** 분류 모델을 여러
    관점에서 평가하는 지표입니다.

## 7. 현재 모델의 범위와 한계

현재 구현된 예측 모델은 주로 **향후 4시간**의 변동성 또는 가격 방향을
예측하도록 구성되어 있습니다. 따라서 이를 1주일 가격 예측 결과로
해석하면 안 됩니다.

모델 결과는 실험적이며 투자 조언이 아닙니다. 특히 가격 방향 분류의
정확도는 제한적이므로 실제 거래 신호로 사용하기 전에 추가 검증이
필요합니다. 변동성을 이용해 표시하는 가격 범위도 검증된 95% 신뢰구간이
아니라 근사치입니다.

## 8. 재현성 및 확인 사항

1.  실행 전에 프로젝트 폴더와 파일 이름이 위 목록과 일치하는지
    확인합니다.
2.  데이터 수집과 대시보드 실행에는 인터넷 연결이 필요할 수 있습니다.
3.  `models/` 폴더에 학습된 모델 파일이 있어야 저장 모델을 사용하는
    기능이 실행됩니다.
4.  실행 오류가 발생하면 오류 메시지와 Python 버전을 기록합니다.
