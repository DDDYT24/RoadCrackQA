# CPU image for scoring photos with a trained model:
#   docker build -t roadcrackqa .
#   then mount the repo's outputs/ at /app/outputs and data/ at /app/data and run
#   batch --input ... --output ... (shell specific commands are in the README).
# The models (under outputs/, see paths.tile_model) are mounted at run time, never baked
# into the image.
FROM python:3.11.14-slim

WORKDIR /app
RUN pip install --no-cache-dir \
    torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs

ENTRYPOINT ["crackqa"]
CMD ["--help"]
