FROM odsai/ecup26-matching-baseline:1.0

WORKDIR /app

RUN python -m pip install --no-cache-dir --break-system-packages --only-binary=:all: \
      "catboost==1.2.10" \
      "transformers==5.0.0" \
      "safetensors==0.7.0" \
 && python -c "import catboost, safetensors, transformers; assert catboost.__version__ == '1.2.10'; assert transformers.__version__ == '5.0.0'; assert safetensors.__version__ == '0.7.0'"
