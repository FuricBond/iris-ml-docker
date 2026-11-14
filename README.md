# Containerize an ML Model Using Docker (Iris Classifier)

MLOps lab assignment (Unit III). A scikit-learn model that predicts the Iris flower
species is trained locally, packaged with its exact dependencies into Docker images,
and served in two ways: as a command-line application (Part A) and as a FastAPI REST
API (Part B).

---

## 1. Aim

To train a machine learning model on the Iris dataset, save it, and containerize it
with Docker together with a prediction application and pinned dependencies, so that the
same model gives the same predictions on any machine with Docker, without installing
Python or any libraries on that machine. As an extension, the model is served as a REST
API from a container and tested through its interactive API documentation and from the
command line.

---

## 2. Technologies and versions

| Component | Version | Used for |
|---|---|---|
| Operating system / shell | Windows, Windows PowerShell 5.1 | Host system; all commands below are PowerShell |
| Python (host, local `venv`) | 3.13.9 | Training the model and local testing |
| Docker | 29.8.2 (Docker Desktop, WSL 2 backend) | Building images and running containers |
| Git | 2.53 | Version control |
| Base image | `python:3.13-slim` (Python 3.13.18 inside the container) | Base of both images |
| scikit-learn | 1.9.1 | Iris dataset, `RandomForestClassifier`, metrics |
| joblib | 1.6.0 | Saving and loading the model |
| numpy | 2.5.4 | Input arrays for prediction |
| fastapi | 0.162.7 | REST API framework (Part B) |
| uvicorn | 0.54.0 | ASGI web server that runs the API (Part B) |
| pydantic | 2.14.3 | Request validation (Part B) |

The library versions are pinned with `==` in `requirements.txt` (Part A) and
`requirements-api.txt` (Part B). They are exactly the versions that trained
`model.joblib`, so the model is loaded by the same scikit-learn version that created it.

---

## 3. Project structure

```
iris-ml-docker/
├── train_model.py         Trains the RandomForest model, prints accuracy + report, saves model.joblib
├── model.joblib           The trained model (167,841 bytes); predicts species names directly
├── app.py                 Part A: command-line prediction app (4 measurements as arguments)
├── api.py                 Part B: FastAPI REST API (GET /, POST /predict, /docs)
├── requirements.txt       Pinned libraries for the CLI image (scikit-learn, joblib, numpy)
├── requirements-api.txt   Pinned libraries for the API image (above + fastapi, uvicorn, pydantic)
├── Dockerfile             Recipe for the CLI image  -> iris-ml-model
├── Dockerfile.api         Recipe for the API image  -> iris-ml-api:1.0
├── .dockerignore          Files Docker must not send to the build (venv, .git, caches)
├── .gitignore             Files Git must not track (venv, caches)
├── README.md              This document
└── venv/                  Local Python virtual environment (not in Git, not in Docker)
```

`venv/` (and Python's `__pycache__/` folders) are listed in both `.gitignore` and
`.dockerignore`. They are never committed and never sent to the Docker build, because
the images install their own libraries from the requirements files.

---

## 4. Part A - Containerized CLI model

### 4.1 Local setup (Windows PowerShell)

Requires Python 3.13 on the host (`python --version`). Start in the folder that contains
`iris-ml-docker` (if you are already inside the project folder, skip the `cd` line):

```powershell
cd iris-ml-docker
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1` because of the execution policy, either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call the
venv interpreter directly, e.g. `.\venv\Scripts\python.exe train_model.py`.

### 4.2 Train the model

```powershell
python train_model.py
```

What the script does:

1. Loads the Iris dataset with `load_iris()` (150 flowers, 4 measurements in cm).
2. Converts the numeric labels 0/1/2 into the species names `setosa`, `versicolor`,
   `virginica`, so the trained model predicts names directly.
3. Splits the data 80/20 into training and testing sets (`test_size=0.2`,
   `random_state=42`, `stratify` so each species has the same share in both sets).
4. Trains `RandomForestClassifier(n_estimators=100, random_state=42)`.
5. Evaluates on the 30 test flowers it has never seen (accuracy + classification report).
6. Saves only the trained model to `model.joblib` next to the script.

Verified output (the absolute path in the last line is shortened here):

```
=======================================================
IRIS MODEL TRAINING
=======================================================
Dataset size      : 150 samples, 4 features
Training samples  : 120
Testing samples   : 30
-------------------------------------------------------
Accuracy          : 0.9000 (90.00%)
-------------------------------------------------------
Classification report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.82      0.90      0.86        10
   virginica       0.89      0.80      0.84        10

    accuracy                           0.90        30
   macro avg       0.90      0.90      0.90        30
weighted avg       0.90      0.90      0.90        30

-------------------------------------------------------
Model saved to    : <project folder>\model.joblib
=======================================================
```

Accuracy 0.90 means 27 of the 30 test flowers were classified correctly. Setosa is
recognised perfectly; the 3 mistakes are between versicolor and virginica, whose
measurements overlap. The resulting `model.joblib` is 167,841 bytes.

### 4.3 Run the CLI app locally

`app.py` reads the four measurements (in cm) from the command line in this order:
sepal length, sepal width, petal length, petal width.

```powershell
python app.py                     # no arguments: default sample 5.1 3.5 1.4 0.2
python app.py 6.7 3.0 5.2 2.3     # a virginica flower from the dataset
python app.py 5.9 3.0 4.2 1.5     # a versicolor flower from the dataset
```

Output format (default sample):

```
Iris Species Prediction
Input:
  Sepal Length : 5.1
  Sepal Width  : 3.5
  Petal Length : 1.4
  Petal Width  : 0.2
Prediction:
  setosa
```

Input rules. Any violation prints an `Error: ...` message to stderr and exits with code 1
(no Python traceback):

| Situation | Behaviour |
|---|---|
| 0 arguments | Uses the default sample 5.1 3.5 1.4 0.2 |
| Number of arguments other than 0 or 4 | Error + usage example |
| Value is not a number (e.g. `abc`) | `... must be a number ...` |
| Value is `nan` or `inf` | `... must be a finite number ...` |
| Value <= 0 or > 20 | `... must be greater than 0 and at most 20 cm ...` |
| `model.joblib` missing | Error telling you to run `python train_model.py` first |

The app takes command-line arguments instead of using `input()` because a container
started with plain `docker run` (without `-it`) has no keyboard input attached.

### 4.4 Build the Docker image

```powershell
docker build -t iris-ml-model .
```

- `-t iris-ml-model` names the image (the tag defaults to `latest`).
- `.` is the build context: the project folder, minus everything in `.dockerignore`.
- Result: image `iris-ml-model:latest`. Only 172.09 kB of build context was transferred:
  BuildKit (Docker's default builder) sends just the files the Dockerfile copies
  (`requirements.txt`, `model.joblib`, `app.py`). `.dockerignore` is the safety net that
  keeps `venv/`, `.git/` and caches out even if a Dockerfile used `COPY . .`.

### 4.5 Run the container

```powershell
# Default command from the Dockerfile: python app.py  (default sample)
docker run --rm iris-ml-model

# Custom input: everything after the image name REPLACES the Dockerfile's CMD
docker run --rm iris-ml-model python app.py 6.7 3.0 5.2 2.3
docker run --rm iris-ml-model python app.py 5.9 3.0 4.2 1.5

# Invalid input (25 cm is out of range) -> error message, exit code 1
docker run --rm iris-ml-model python app.py 5.1 25 1.4 0.2
$LASTEXITCODE
```

`--rm` deletes the container automatically when it exits, so finished test runs do not
pile up.

| Command | Verified result |
|---|---|
| `docker run --rm iris-ml-model` | `setosa` (default sample 5.1 3.5 1.4 0.2) |
| `... python app.py 6.7 3.0 5.2 2.3` | `virginica` |
| `... python app.py 5.9 3.0 4.2 1.5` | `versicolor` |
| `... python app.py 5.1 25 1.4 0.2` | Range error message, exit code 1 |

Checks that prove the model really runs inside the container and not on the host:

| Check | Verified result |
|---|---|
| `platform.system()` inside the container | `Linux` (the host is Windows) |
| scikit-learn version inside the container | 1.9.1 |
| Contents of `/app` in the container | Only `app.py`, `model.joblib`, `requirements.txt` |
| Host venv deactivated (`deactivate`) before `docker run` | Predictions still worked |

These can be reproduced with, for example:

```powershell
docker run --rm iris-ml-model python -c "import platform, sklearn; print(platform.system(), sklearn.__version__)"
docker run --rm iris-ml-model ls /app
```

### 4.6 Verify with Docker commands

```powershell
docker images                                  # list images
docker ps                                      # running containers
docker ps -a                                   # all containers, including stopped ones

# A named run WITHOUT --rm, to see the container afterwards (any name works)
docker run --name iris-cli-test iris-ml-model
docker ps -a                                   # shows iris-cli-test with status Exited (0)
docker logs iris-cli-test                      # shows the prediction output again
docker rm iris-cli-test                        # delete the stopped container
```

Observed:

- `docker images` lists `iris-ml-model:latest` with disk usage 548MB and content size 128MB.
- `docker ps` is empty after the `--rm` runs. The CLI container is a batch job: it makes
  one prediction and exits immediately, and `--rm` then deletes it.
- The named run without `--rm` appears in `docker ps -a` as `Exited (0)` (finished
  successfully). `docker logs` shows its prediction. `docker rm` removes it.

---

## 5. Part B - Containerized FastAPI REST API

### 5.1 Endpoints

| Method | Path | Purpose | Response |
|---|---|---|---|
| GET | `/` | Check that the API is running | `{"message": "Iris Model API is running", "docs": "/docs"}` |
| POST | `/predict` | Predict the species for four measurements | `{"prediction": "setosa"}` |
| GET | `/docs` | Interactive Swagger UI, generated automatically by FastAPI | HTML page |

The model is loaded once, when the server starts (FastAPI lifespan handler), and reused
for every request. If `model.joblib` is missing, startup fails immediately with a clear
error instead of failing on every request.

### 5.2 Input validation

The request body must be a JSON object with four numbers (in cm). Pydantic checks each
one: it must be **greater than 0 and at most 20**, the same rule as the CLI app.

| Request problem | HTTP status |
|---|---|
| A value is out of range (e.g. `sepal_width` = 25) | 422 Unprocessable Content |
| A field is missing (e.g. no `petal_width`) | 422 |
| A value is not a number (e.g. `"abc"`) | 422 |
| Malformed JSON | 422 |
| The model fails during prediction | 500 with `{"detail": "Prediction failed"}` (no internal details) |

The 422 responses are produced automatically by FastAPI and Pydantic, from the field
definitions (required `float` fields with `gt=0` and `le=20`) and from FastAPI's JSON
parsing; the API has no custom error handlers.

### 5.3 Request and response examples

Valid request to `POST /predict`:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response (HTTP 200):

```json
{"prediction": "setosa"}
```

Invalid request (`sepal_width` = 25), response (HTTP 422):

```json
{"detail":[{"type":"less_than_equal","loc":["body","sepal_width"],"msg":"Input should be less than or equal to 20","input":25,"ctx":{"le":20.0}}]}
```

### 5.4 Optional: run the API locally without Docker

```powershell
pip install -r requirements-api.txt
python -m uvicorn api:app
```

Then open <http://127.0.0.1:8000/docs>. Stop the server with `Ctrl+C`. Locally, Uvicorn
listens only on 127.0.0.1 (this computer), which is fine outside Docker.

### 5.5 Build and run the API container

```powershell
docker build -f Dockerfile.api -t iris-ml-api:1.0 .
docker run -d -p 8000:8000 --name iris-api-container iris-ml-api:1.0
docker ps
```

- `-f Dockerfile.api` selects the API recipe (the default file name is `Dockerfile`).
- `-t iris-ml-api:1.0` gives the image a name and an explicit version tag.
- `-d` runs the container in the background (detached). The API is a long-running
  server, unlike the CLI batch job.
- `-p 8000:8000` publishes the port: host port 8000 -> container port 8000.
- `--name iris-api-container` gives the container a fixed name for later commands.

Observed:

- `docker images` lists `iris-ml-api:1.0` with disk usage 572MB and content size 134MB.
- During the build, the base image layer and the `WORKDIR /app` layer were reused from
  the cache created by the CLI image (see section 7.3).
- `docker ps` shows `iris-api-container` with status `Up` and ports `0.0.0.0:8000->8000/tcp`.

### 5.6 Test the API

#### a) Swagger UI (API docs)

1. Open <http://localhost:8000/docs> in a browser.
2. Expand **POST /predict** and click **Try it out**. The request body is pre-filled
   with the example 5.1, 3.5, 1.4, 0.2.
3. Click **Execute**.

Verified result: response code 200, body `{"prediction": "setosa"}`.

#### b) PowerShell `Invoke-RestMethod`

```powershell
$body = @{
    sepal_length = 5.9
    sepal_width  = 3.0
    petal_length = 4.2
    petal_width  = 1.5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

`ConvertTo-Json` builds the JSON body for you, so no manual quoting is needed.
Verified result: `versicolor`. PowerShell prints it as a small table:

```
prediction
----------
versicolor
```

#### c) `curl.exe`

In **Windows PowerShell 5.1**, `curl` is not the real curl program: it is an alias of
`Invoke-WebRequest`, which has different parameters. That is why
`curl -X POST ...` fails with `A parameter cannot be found that matches parameter name 'X'`.
Type `curl.exe` to run the real curl that ships with Windows 10/11.

Windows PowerShell 5.1 also does not escape double quotes inside an argument when it
passes it to a native program like `curl.exe`, so `curl.exe` receives the JSON with its
quotes stripped. The JSON must therefore be written in single quotes with every inner
double quote escaped as `\"`. Without the escapes, the
server receives broken JSON and answers 422 `JSON decode error`.

```powershell
# Valid request -> 200
curl.exe -i -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{\"sepal_length\": 6.7, \"sepal_width\": 3.0, \"petal_length\": 5.2, \"petal_width\": 2.3}'

# Out-of-range request (sepal_width = 25) -> 422
curl.exe -i -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{\"sepal_length\": 5.1, \"sepal_width\": 25, \"petal_length\": 1.4, \"petal_width\": 0.2}'
```

`-i` prints the HTTP status line and headers before the body. Verified results:

| Request | Status line | Body |
|---|---|---|
| 6.7, 3.0, 5.2, 2.3 | `HTTP/1.1 200 OK` | `{"prediction":"virginica"}` |
| `sepal_width` = 25 | `HTTP/1.1 422 Unprocessable Content` | `less_than_equal` error for `sepal_width` (limit `le` 20) |

This `\"` escaping is needed in Windows PowerShell 5.1, the shell used and tested here
(PowerShell 7.0-7.2 behave the same way). PowerShell 7.3 and later pass arguments to
native programs exactly as written, so there the backslashes must be left out; otherwise
`curl.exe` sends them to the server, which answers 422 `JSON decode error`.

#### d) Container logs

```powershell
docker logs iris-api-container
```

The Uvicorn access log shows each request with client address `172.17.0.1`, the gateway
of Docker's default bridge network. Requests from the browser or PowerShell reach the
container through Docker's port forwarding, so the container sees the bridge gateway
as the client, not the Windows host.

### 5.7 Stop, restart and inspect the container

```powershell
docker stop iris-api-container       # graceful stop
docker ps -a                         # status: Exited (0)
docker start iris-api-container      # start the SAME container again
docker exec -it iris-api-container bash   # open a shell inside the running container
```

Inside the container shell, `pwd`, `python --version` and `pip list` show the working
directory, Python version and installed libraries; type `exit` to leave.

Observed:

- `docker stop`: status `Exited (0)`. The logs show a graceful Uvicorn shutdown, and a
  request with curl failed with "connection refused" while the container was stopped.
- `docker start`: the same container ID came back `Up`, and curl worked again.
- `docker exec -it ... bash`: working directory `/app`, Python 3.13.18, and all pinned
  library versions installed.

Tagging and removing a tag:

```powershell
docker tag iris-ml-api:1.0 iris-ml-api:latest   # second name for the same image
docker rmi iris-ml-api:latest                   # remove only that name
```

Observed: `docker rmi iris-ml-api:latest` printed only `Untagged`. No layers were
deleted, because the same image is still referenced by the `iris-ml-api:1.0` tag.

#### Cleanup (when finished)

```powershell
docker stop iris-api-container
docker rm iris-api-container
```

Images can be removed with `docker rmi iris-ml-api:1.0` and `docker rmi iris-ml-model`.

#### Optional: push to Docker Hub (not performed)

The optional Docker Hub step from the lab manual was **not** performed in this project.
It would look like this, with your own Docker Hub user name:

```powershell
docker login
docker tag iris-ml-api:1.0 <dockerhub-username>/iris-ml-api:1.0
docker push <dockerhub-username>/iris-ml-api:1.0
```

---

## 6. Docker management commands used

| Command | Purpose | Observed result |
|---|---|---|
| `docker build -t iris-ml-model .` | Build the CLI image from `Dockerfile` | `iris-ml-model:latest`; build context 172.09 kB |
| `docker build -f Dockerfile.api -t iris-ml-api:1.0 .` | Build the API image from `Dockerfile.api` | `iris-ml-api:1.0`; base and `WORKDIR` layers reused from cache |
| `docker images` | List local images | `iris-ml-model:latest` 548MB / 128MB; `iris-ml-api:1.0` 572MB / 134MB (disk usage / content size) |
| `docker run --rm iris-ml-model` | Run one prediction, then delete the container | `setosa` |
| `docker run --rm iris-ml-model python app.py 6.7 3.0 5.2 2.3` | Run with custom input (overrides `CMD`) | `virginica` |
| `docker run --rm iris-ml-model python app.py 5.1 25 1.4 0.2` | Check input validation in the container | Range error, exit code 1 |
| `docker ps` | List running containers | Empty after `--rm` runs; `iris-api-container` `Up` with `0.0.0.0:8000->8000/tcp` |
| `docker ps -a` | List all containers, including stopped ones | Named CLI run shown as `Exited (0)` |
| `docker logs <container>` | Show a container's output | CLI: the prediction; API: Uvicorn access log (client `172.17.0.1`) |
| `docker rm <container>` | Delete a stopped container | Named CLI test container removed |
| `docker run -d -p 8000:8000 --name iris-api-container iris-ml-api:1.0` | Start the API in the background with a published port | Container `Up`; API reachable on port 8000 |
| `docker stop iris-api-container` | Stop gracefully | `Exited (0)`; graceful Uvicorn shutdown; curl: connection refused |
| `docker start iris-api-container` | Restart the same stopped container | Same container ID, `Up` again; curl works |
| `docker exec -it iris-api-container bash` | Open a shell inside the running container | `/app`, Python 3.13.18, pinned versions present |
| `docker tag iris-ml-api:1.0 iris-ml-api:latest` | Add a second name (tag) to an image | New tag pointing to the same image |
| `docker rmi iris-ml-api:latest` | Remove an image tag | Only `Untagged`; image kept because of the `1.0` tag |
| `docker push ...` | Upload the image to Docker Hub | Not performed |

---

## 7. Dockerfiles explained

### 7.1 `Dockerfile` (Part A, CLI image `iris-ml-model`)

| Instruction | Meaning |
|---|---|
| `FROM python:3.13-slim` | Start from the official slim Debian-based Python 3.13 image, the same Python version that trained the model |
| `WORKDIR /app` | Create `/app` and use it as the working directory for the following instructions and for the running container |
| `COPY requirements.txt .` | Copy only the dependency list first (for layer caching, see 7.3) |
| `RUN pip install --no-cache-dir -r requirements.txt` | Install the pinned libraries; `--no-cache-dir` stops pip from keeping its download cache, which keeps the image smaller |
| `COPY model.joblib .` | Copy the trained model into `/app` |
| `COPY app.py .` | Copy the prediction app into `/app` |
| `CMD ["python", "app.py"]` | Default command when a container starts; replaced by any command given after the image name in `docker run` |

### 7.2 `Dockerfile.api` (Part B, API image `iris-ml-api:1.0`)

| Instruction | Meaning |
|---|---|
| `FROM python:3.13-slim` | Same base image as the CLI image |
| `WORKDIR /app` | Same working directory |
| `COPY requirements-api.txt .` | Copy the API dependency list first (layer caching) |
| `RUN pip install --no-cache-dir -r requirements-api.txt` | Install the pinned ML libraries plus fastapi, uvicorn and pydantic |
| `COPY model.joblib .` | Copy the trained model |
| `COPY api.py .` | Copy the API code |
| `EXPOSE 8000` | Document that the server inside listens on port 8000 |
| `CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]` | Start Uvicorn serving the `app` object from `api.py` on all network interfaces, port 8000 |

### 7.3 Layer caching: why the requirements file is copied first

Each instruction creates an image layer. When an image is rebuilt, Docker reuses a
cached layer if that instruction and the files it copies have not changed. As soon as
one layer changes, every layer after it is rebuilt.

The `pip install` step is the slowest one. Because the requirements file is copied and
installed before the code, editing `app.py` or `api.py` only invalidates the final
`COPY` layer. The library installation is reused from the cache. If the code were copied
first, every code change would trigger a full reinstall of all libraries.

Observed: when `iris-ml-api:1.0` was built after `iris-ml-model`, the base image layer
and the `WORKDIR /app` layer were reused from the cache, because both Dockerfiles start
with the same two instructions. The `pip install` layer was not shared, because
`requirements-api.txt` differs from `requirements.txt`.

### 7.4 `EXPOSE` vs `-p`

`EXPOSE 8000` is documentation (image metadata). It tells users which port the
application uses but does **not** make it reachable from the host. The port is actually
published by `docker run -p 8000:8000` (format `HOST_PORT:CONTAINER_PORT`). Without
`-p`, the API would run but could not be reached from the browser on Windows.

### 7.5 Why `--host 0.0.0.0`

By default Uvicorn listens on `127.0.0.1`, the loopback address. Inside a container,
that means "only this container itself". Traffic forwarded by `-p` arrives on the
container's network interface, not on its loopback, so it would be refused.
`--host 0.0.0.0` makes Uvicorn listen on all interfaces of the container, so the
published port works.

### 7.6 Exec-form `CMD` and graceful shutdown

Both Dockerfiles use the exec form, a JSON array such as
`CMD ["uvicorn", "api:app", ...]`. The program then runs directly as the container's
main process (PID 1) and receives the stop signal (SIGTERM) sent by `docker stop`.
Uvicorn reacts by shutting down gracefully, which is why the container stopped with
`Exited (0)` and the logs showed a clean shutdown. With the shell form
(`CMD uvicorn api:app ...`) the command is started through `/bin/sh -c`, and the shell may
become PID 1 and not pass the signal on. Docker then waits for its stop timeout
(10 seconds by default) and force-kills the process, with no clean shutdown. The exec
form avoids this.

---

## 8. Design decisions and deviations from the handouts

Two handouts were followed: **PDF A**, "Containerize an ML Model Using Docker" (Unit III),
and **PDF B**, "MLOps Lab: Containerizing an ML Model using Docker" (T.Y. B.Tech AI&DS
lab manual). Where they differ, or where a change was needed, this project did the following:

| Handout says | This project | Why |
|---|---|---|
| Base image `python:3.11-slim` / `python:3.9-slim` | `python:3.13-slim` | Matches the Python version that trained the model; the pinned library versions have ready-made wheels for Python 3.13 |
| Unpinned libraries, or PDF B's pins (scikit-learn 1.5.1, numpy 1.26.4) | Pinned to scikit-learn 1.9.1, joblib 1.6.0, numpy 2.5.4 (+ fastapi 0.162.7, uvicorn 0.54.0, pydantic 2.14.3 for the API) | PDF B's pins are not installable on Python 3.13; the model must be loaded by the same library versions that trained it |
| Flask API on port 5000 (PDF B) | FastAPI on port 8000 (PDF A extension) | Automatic Swagger docs at `/docs` and automatic input validation with Pydantic |
| Image `iris-model-api:1.0`, container `iris-container` (PDF B) | Images `iris-ml-model` and `iris-ml-api:1.0`, container `iris-api-container` | Follows PDF A's names (`iris-ml-model`, `iris-ml-api`) while keeping PDF B's version tag `1.0` |
| Model saved as `iris_model.pkl` | `model.joblib` (PDF A) | Follows PDF A; joblib stores scikit-learn models with large NumPy arrays efficiently |
| Prediction returned as a class number, e.g. `0` (PDF B) | Species name, e.g. `"setosa"` (PDF A) | The model is trained on name labels, so the output is readable without a lookup table |
| Train on the full dataset without evaluation (PDF B) | 80/20 stratified train/test split with accuracy and classification report | PDF A requires evaluation, and a model should be measured on data it has not seen |
| `app.py` accepts four input features (PDF A) | Features read from command-line arguments, with a default sample when none are given | `docker run` without `-it` has no keyboard input; the default sample makes plain `docker run --rm iris-ml-model` work |
| (Not specified) | Separate `Dockerfile.api` and `requirements-api.txt` | Part A stays unchanged, and the CLI image does not carry web-server libraries |
| (Not specified) | `train_model.py` is not copied into either image | Containers only serve the model; training happens before the build |
| (Not specified) | `.dockerignore` excludes `venv/`, `.git/` and caches | Guarantees these are never sent to the builder, even if a Dockerfile used `COPY . .`, so the host venv can never end up in an image |

---

## 9. Results summary

| Item | Result |
|---|---|
| Test accuracy (30 test flowers) | 0.9000 (90.00%); setosa 100% correct |
| Model file | `model.joblib`, 167,841 bytes |
| CLI image `iris-ml-model:latest` | 548MB disk usage / 128MB content size |
| API image `iris-ml-api:1.0` | 572MB disk usage / 134MB content size |
| CLI container predictions | 5.1 3.5 1.4 0.2 -> setosa; 6.7 3.0 5.2 2.3 -> virginica; 5.9 3.0 4.2 1.5 -> versicolor |
| CLI invalid input (25 cm) | Error message, exit code 1 |
| API via Swagger UI | 200 `{"prediction": "setosa"}` |
| API via `Invoke-RestMethod` | `versicolor` |
| API via `curl.exe` | 200 `virginica`; out-of-range value -> 422 |
| Container stop / start | Graceful shutdown with `Exited (0)`; same container restarted and served again |
| Runs on a Linux container from a Windows host | Yes (`platform.system()` = `Linux`) |

---

## 10. Limitations

- **NaN / Infinity in the API.** A JSON body containing `NaN`, `Infinity` or a number too
  large for a float (e.g. `1e309`) gets a plain `500 Internal Server Error` instead of
  422. Pydantic does reject the value, but FastAPI's built-in 422 handler cannot write
  `inf`/`nan` back into its JSON error message. Standard JSON does not allow `NaN` or
  `Infinity` anyway. This was accepted and not fixed.
- **Booleans in the API.** JSON `true` is accepted as `1.0`, because Pydantic converts
  compatible types by default (lax mode).
- **Small test set.** The accuracy is measured on only 30 flowers, so 90% is a rough
  estimate. One more or one fewer mistake changes it by about 3 percentage points.
- **Model baked into the image.** `model.joblib` is copied into the image at build time,
  so a retrained model requires rebuilding the image.
- **Not production-ready.** The API runs a single Uvicorn worker, with no authentication
  and no HTTPS. It is intended only for local use in this lab.

---

## 11. Post-lab (viva) questions

**Q1. What is the difference between a Docker image and a Docker container?**

An image is a read-only template made of stacked layers: the base OS files, Python, the
installed libraries, and our `model.joblib` and code. A container is a running instance
of an image. Docker adds a thin writable layer on top of the image layers, and anything
written at runtime goes there. One image can start many containers: every
`docker run --rm iris-ml-model ...` above created a new, separate container from the same
`iris-ml-model` image. When the API container imports `api.py`, Python creates a
`__pycache__` folder in `/app`. That folder lives only in the container's writable layer;
the image itself is unchanged.

**Q2. How is Docker different from a virtual machine?**

A virtual machine emulates a whole computer and runs its own complete operating system
kernel, so it typically needs gigabytes of disk and takes a while to boot. Containers
share the host's kernel (on Windows, the Linux kernel of the WSL 2 backend) and only
package the application and its user-space files. Our images are 128MB and 134MB of
content, and a container starts in seconds. The trade-off is weaker isolation:
containers are isolated processes on a shared kernel, while a VM has a separate kernel.

**Q3. What is the role of a Dockerfile? Explain FROM, WORKDIR, COPY, RUN, EXPOSE and CMD.**

A Dockerfile is a text recipe that builds an image step by step, so the environment is
reproducible and version-controlled together with the code.

- `FROM python:3.13-slim` chooses the base image.
- `WORKDIR /app` sets the working directory.
- `COPY` copies files from the build context into the image: `requirements.txt` (or
  `requirements-api.txt`), then `model.joblib` and `app.py` (or `api.py`).
- `RUN pip install --no-cache-dir -r requirements.txt` (or `-r requirements-api.txt`)
  executes a command at build time and saves the result as a layer.
- `EXPOSE 8000` (API only) documents the port the server listens on.
- `CMD` sets the default command at container start: `["python", "app.py"]` for the CLI
  and `["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]` for the API.

**Q4. What is layer caching, and why is the requirements file copied before the code?**

Every Dockerfile instruction produces a layer. On a rebuild, Docker reuses a layer from
the cache if the instruction and its input files are unchanged; after the first changed
layer, all following layers are rebuilt. Copying `requirements.txt` (or
`requirements-api.txt`) and running `pip install` before copying the code means that a
change to `app.py` or `api.py` only rebuilds the last `COPY` step. The slow library
installation is reused. We saw layer caching in action when building `iris-ml-api:1.0`:
the base layer and the `WORKDIR /app` layer were taken from the cache created by the CLI
image, while the `pip install` layer was rebuilt because `requirements-api.txt` differs
from `requirements.txt`.

**Q5. How can logs or data be kept after a container is removed?**

Anything written inside a container (its writable layer), including the output shown by
`docker logs`, is deleted by `docker rm`. To persist data, use a **volume**, which Docker
manages and which outlives containers, e.g. `docker volume create iris-logs` and
`docker run -v iris-logs:/app/logs ...`. Alternatively, use a **bind mount**, which maps
a host folder into the container, e.g. `-v "${PWD}\logs:/app/logs"` in PowerShell (the
quotes matter if the path contains spaces). The app
would then write its log files to that path. Our project writes no log files: all its
output goes to the console (stdout and stderr), which `docker logs` shows, so it
disappears with the container.

**Q6. How would Docker fit into a CI/CD pipeline for ML?**

A commit to the Git repository triggers the CI pipeline. It installs the pinned
requirements, trains the model and runs the evaluation and tests. If they pass, it runs
`docker build` and tags the image with a version (as we did with `:1.0`), then pushes it
to a registry such as Docker Hub. The CD stage deploys exactly that immutable image to a
test environment and then to production, for example a cloud service or a Kubernetes
cluster. Because the same image runs everywhere, "it works on my machine" problems
disappear, and rolling back means redeploying the previous tag.

**Q7. How can a container use GPUs for deep learning?**

By default containers do not see the host's GPUs. The host needs an NVIDIA driver and the
NVIDIA Container Toolkit, and the container is started with `docker run --gpus all ...`.
The image is usually built from a CUDA base image (e.g. an `nvidia/cuda` or a
framework's GPU image), and the CUDA version inside must be compatible with the host
driver. GPU images are large, often several gigabytes. Our project does not need any of
this: `RandomForestClassifier` runs on the CPU only.

**Q8. How can the containerized model API be scaled to handle more requests?**

Inside one container, Uvicorn can run several worker processes (`--workers N`), or
Gunicorn can manage Uvicorn workers; each worker loads its own copy of the model. Across
containers, several replicas of the same image can run behind a load balancer, for
example with Docker Compose. In Kubernetes, a Deployment sets the number of replicas, and
a Horizontal Pod Autoscaler adds or removes replicas based on load. This is easy for our
API because it is stateless: the model is read-only and loaded at startup, and no request
depends on a previous one, so any replica can answer any request.

---

## 12. Conclusion

A RandomForest classifier was trained on the Iris dataset (90% accuracy on a held-out
test set) and saved as `model.joblib`. It was then containerized twice from the same
model file: as a command-line batch image (`iris-ml-model`) that predicts from
command-line arguments, and as a FastAPI REST API image (`iris-ml-api:1.0`) with
automatic input validation and Swagger documentation. Pinning both the Python version
(`python:3.13-slim`) and the library versions to the training environment made the
containers give the same predictions as the local setup, on Linux containers running
on a Windows host, without the host's virtual environment. The lab also covered the
everyday Docker workflow: building, running, inspecting, logging, stopping, restarting,
tagging and cleaning up images and containers. These are the basic steps for shipping ML
models reproducibly in an MLOps pipeline.
