document.addEventListener("DOMContentLoaded", () => {
    const candidateForm = document.getElementById("candidateForm");
    const candidateSection = document.getElementById("candidateSection");
    const interviewSection = document.getElementById("interviewSection");

    const cameraPreview = document.getElementById("cameraPreview");
    const cameraOverlay = document.getElementById("cameraOverlay");
    const cameraStatus = document.getElementById("cameraStatus");
    const microphoneStatus = document.getElementById("microphoneStatus");
    const faceStatus = document.getElementById("faceStatus");

    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("status");
    const timerEl = document.getElementById("timer");

    const violationCountEl = document.getElementById("violationCount");
    const riskScoreEl = document.getElementById("riskScore");
    const eventLog = document.getElementById("eventLog");

    const chatBox = document.getElementById("chat");
    const speakBtn = document.getElementById("startBtn");
    const endBtn = document.getElementById("stopBtn");

    // Keep this URL for local development; change it to your deployed API URL later.
const API_BASE_URL = "https://ai-voice-recruiter-4dgj.onrender.com";
    let mediaStream = null;
    let interviewStartTime = null;
    let timerInterval = null;
    let violationCount = 0;
    let currentQuestion = "";
    let candidateData = {};

    let interviewEnded = false;
    let evaluationInProgress = false;
    let monitoringInitialized = false;

    // Face monitoring checks visible faces only.
    // It cannot determine honesty or prove cheating.
    let faceLandmarker = null;
    let faceMonitorInterval = null;
    let noFaceFrames = 0;
    let multipleFaceFrames = 0;
    let lastFaceWarningAt = { noFace: 0, multiple: 0 };
    let faceViolationCount = 0;

    // Face-monitoring warnings never automatically terminate an interview.
    const FACE_WARNING_COOLDOWN_MS = 10000;

    let sustainedLookAwaySince = null;
    let lastGazeWarningAt = 0;

    const LOOK_AWAY_HOLD_MS = 8000;
    const GAZE_WARNING_COOLDOWN_MS = 30000;

    const answerHistory = [];

    // AI voice output
    function speakText(text) {
        if (!("speechSynthesis" in window) || !text || interviewEnded) {
            return;
        }

        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "en-US";
        utterance.rate = 1;
        utterance.pitch = 1;

        window.speechSynthesis.speak(utterance);
    }

    function stopVoiceOutput() {
        if ("speechSynthesis" in window) {
            window.speechSynthesis.cancel();
        }
    }

    // Monitoring events
    function logEvent(message, isError = false) {
        violationCount++;

        if (violationCountEl) {
            violationCountEl.textContent = violationCount;
        }

        if (riskScoreEl) {
            riskScoreEl.textContent =
                `${Math.min(violationCount * 15, 100)}/100`;
        }

        if (!eventLog) return;

        if (eventLog.querySelector(".empty-log")) {
            eventLog.innerHTML = "";
        }

        const p = document.createElement("p");
        p.textContent =
            `[${new Date().toLocaleTimeString()}] ${message}`;

        if (isError) {
            p.style.color = "#ef4444";
        }

        eventLog.prepend(p);
    }

    function recordFaceViolation(message, kind) {
        if (interviewEnded) return;

        const now = Date.now();

        if (now - lastFaceWarningAt[kind] < FACE_WARNING_COOLDOWN_MS) {
            return;
        }

        lastFaceWarningAt[kind] = now;
        faceViolationCount++;

        logEvent(message, true);
        const liveAnswer = `${finalTranscript} ${interimTranscript}`.trim();

statusText.textContent = liveAnswer
    ? `Listening: ${liveAnswer}`
    : "Listening to your answer...";
        statusText.textContent =
            `Monitoring warning (${faceViolationCount}): ${message} ` +
            "This is an imperfect automated signal, not proof of misconduct.";
    }

    async function initFaceMonitoring() {
        const vision =
            window.vision || window.MediaPipeTasksVision;

        if (!vision?.FaceLandmarker) {
            faceStatus.textContent =
                "Face detection: Library unavailable";

            logEvent(
                "Face detection library did not load. Camera is active, but face detection is unavailable.",
                true
            );

            return;
        }

        try {
            const fileset =
                await vision.FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
                );

            faceLandmarker =
                await vision.FaceLandmarker.createFromOptions(
                    fileset,
                    {
                        baseOptions: {
                            modelAssetPath:
                                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                            delegate: "GPU"
                        },
                        runningMode: "VIDEO",
                        numFaces: 2
                    }
                );

            if (interviewEnded) {
                faceLandmarker.close();
                faceLandmarker = null;
                return;
            }

            faceStatus.textContent = "Face detection: Running";
            logEvent("Face-presence monitoring started.");

            faceMonitorInterval = setInterval(() => {
                if (
                    interviewEnded ||
                    !faceLandmarker ||
                    cameraPreview.readyState < 2
                ) {
                    return;
                }

                try {
                    const result = faceLandmarker.detectForVideo(
                        cameraPreview,
                        performance.now()
                    );

                    const count =
                        result.faceLandmarks?.length ?? 0;

                    if (count === 0) {
                        noFaceFrames++;
                        multipleFaceFrames = 0;

                        if (noFaceFrames >= 4) {
                            noFaceFrames = 0;

                            recordFaceViolation(
                                "No face detected for several checks. Ensure your face is visible to the camera.",
                                "noFace"
                            );
                        }
                    } else if (count > 1) {
                        multipleFaceFrames++;
                        noFaceFrames = 0;

                        if (multipleFaceFrames >= 3) {
                            multipleFaceFrames = 0;

                            recordFaceViolation(
                                "More than one face detected. Please ensure only the interview candidate is in frame.",
                                "multiple"
                            );
                        }
                    } else {
                        noFaceFrames = 0;
                        multipleFaceFrames = 0;

                        faceStatus.textContent =
                            "Face detection: One face visible";

                        const lookingAway =
                            estimateLookingAway(
                                result.faceLandmarks[0]
                            );

                        const now = Date.now();

                        if (lookingAway) {
                            if (sustainedLookAwaySince === null) {
                                sustainedLookAwaySince = now;
                            }

                            if (
                                now - sustainedLookAwaySince >= LOOK_AWAY_HOLD_MS &&
                                now - lastGazeWarningAt >= GAZE_WARNING_COOLDOWN_MS
                            ) {
                                lastGazeWarningAt = now;
                                sustainedLookAwaySince = null;

                                logEvent(
                                    "Sustained sideways-gaze estimate. This is only a review signal; looking away is not proof of cheating.",
                                    false
                                );

                                faceStatus.textContent =
                                    "Face detection: Face visible; gaze review signal";
                            }
                        } else {
                            sustainedLookAwaySince = null;
                        }
                    }
                } catch (err) {
                    console.warn(
                        "Face detection frame failed:",
                        err
                    );
                }
            }, 500);
        } catch (err) {
            console.error(
                "Could not initialize face detection:",
                err
            );

            faceStatus.textContent =
                "Face detection: Unavailable";

            logEvent(
                "Face detector could not initialize. Check internet access and reload the page.",
                true
            );
        }
    }

    function estimateLookingAway(faceLandmarks) {
        // This is only an approximate gaze signal.
        if (!faceLandmarks || faceLandmarks.length < 478) {
            return false;
        }

        const ids = [468, 473, 133, 33, 362, 263];
        const pts = ids.map(i => faceLandmarks[i]);

        if (!pts.every(Boolean)) return false;

        const [
            leftIris,
            rightIris,
            leftInner,
            leftOuter,
            rightInner,
            rightOuter
        ] = pts;

        const leftWidth =
            Math.abs(leftInner.x - leftOuter.x);

        const rightWidth =
            Math.abs(rightInner.x - rightOuter.x);

        if (leftWidth < 0.001 || rightWidth < 0.001) {
            return false;
        }

        const leftRatio =
            (leftIris.x - Math.min(leftInner.x, leftOuter.x)) /
            leftWidth;

        const rightRatio =
            (rightIris.x - Math.min(rightInner.x, rightOuter.x)) /
            rightWidth;

        const average = (leftRatio + rightRatio) / 2;

        return average < 0.20 || average > 0.80;
    }

    // Candidate form and hardware permissions
    candidateForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        if (interviewSection.hidden === false) return;

        const name =
            document.getElementById("candidateName").value.trim();

        const email =
            document.getElementById("candidateEmail").value.trim();

        const jobRole =
            document.getElementById("jobRole").value;

        const consentCheckbox =
            document.getElementById("monitoringConsent");

        if (!consentCheckbox.checked) {
            alert("You must check the consent box to proceed.");
            consentCheckbox.focus();
            return;
        }

        if (!name || !email || !jobRole) {
            alert(
                "Please enter your name, email, and select a job role."
            );
            return;
        }

        candidateData = { name, email, jobRole };

        statusText.textContent =
            "Requesting camera and microphone access...";

        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error(
                    "Camera and microphone access is not supported here. " +
                    "Open the page through http://127.0.0.1:5500/index.html"
                );
            }

            mediaStream =
                await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 },
                    audio: true
                });

            if (interviewEnded) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
                return;
            }

            cameraPreview.srcObject = mediaStream;
            cameraPreview.play().catch(() => {});

            cameraOverlay.hidden = true;

            cameraStatus.textContent = "Camera: Active";
            cameraStatus.style.background = "#d1fae5";
            cameraStatus.style.color = "#065f46";

            microphoneStatus.textContent = "Mic: Active";
            microphoneStatus.style.background = "#d1fae5";
            microphoneStatus.style.color = "#065f46";

            
        } catch (err) {
            console.error("Hardware permission error:", err);

            alert(
                "Camera and microphone access is required. " +
                "Please allow access in your browser and try again.\n\n" +
                err.message
            );

            cameraStatus.textContent = "Camera: Failed";
            microphoneStatus.textContent = "Mic: Failed";
            faceStatus.textContent = "Face: Failed";

            if (cameraOverlay) {
                cameraOverlay.hidden = false;
                cameraOverlay.textContent =
                    "Camera or microphone permission failed.";
            }

            return;
        }

        candidateSection.hidden = true;
        interviewSection.hidden = false;

        statusDot.classList.add("active");

        statusText.textContent =
            "Hardware verified. Connecting to AI Recruiter...";

        startTimer();
        initMonitoringEvents();


        if (interviewEnded) return;

        await startAIInterview();
    });

    // Interview timer
    function startTimer() {
        interviewStartTime = Date.now();

        timerInterval = setInterval(() => {
            const elapsedSeconds = Math.floor(
                (Date.now() - interviewStartTime) / 1000
            );

            const mins = String(
                Math.floor(elapsedSeconds / 60)
            ).padStart(2, "0");

            const secs = String(
                elapsedSeconds % 60
            ).padStart(2, "0");

            timerEl.textContent = `⏱ ${mins}:${secs}`;
        }, 1000);
    }

    // Monitoring
    function initMonitoringEvents() {
        if (monitoringInitialized) return;

        monitoringInitialized = true;

        document.addEventListener("visibilitychange", () => {
            if (document.hidden && !interviewEnded) {
                logEvent(
                    "Candidate switched tabs or minimized window.",
                    true
                );
            }
        });

        window.addEventListener("blur", () => {
            if (!interviewEnded) {
                logEvent(
                    "Candidate window lost focus.",
                    true
                );
            }
        });
    }

    // Start the AI interview
    async function startAIInterview() {
        try {
            const response = await fetch(
                `${API_BASE_URL}/start-interview`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(candidateData)
                }
            );

            if (interviewEnded) return;

            if (!response.ok) {
                throw new Error(
                    `Interview request failed: HTTP ${response.status}`
                );
            }

            const data = await response.json();

            if (interviewEnded) return;

            if (!data || !data.question) {
                throw new Error(
                    "Invalid response from AI backend."
                );
            }

            currentQuestion = data.question;

            appendMessage("AI", currentQuestion);
            speakText(currentQuestion);

            statusText.textContent =
                "Interview ready. Click Speak when you are ready to answer.";

            speakBtn.disabled = false;
        } catch (err) {
            if (interviewEnded) return;

            console.error(
                "Failed to start AI interview:",
                err
            );

            currentQuestion =
                `Hello ${candidateData.name}! Let's start with your ` +
                `background in ${candidateData.jobRole}. ` +
                "Can you describe a challenging technical project you built?";

            appendMessage("AI", currentQuestion);
            speakText(currentQuestion);

            statusText.textContent =
                "Fallback question loaded. Click Speak to answer.";

            speakBtn.disabled = false;
        }
    }

    // Add messages to the interview chat
    function appendMessage(sender, text) {
        if (interviewEnded && sender !== "System") return;

        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender.toLowerCase()}`;

        const strong = document.createElement("strong");

        strong.textContent =
            sender === "AI"
                ? "AI Recruiter"
                : sender === "User"
                    ? candidateData.name
                    : "System";

        msgDiv.appendChild(strong);
        msgDiv.appendChild(
            document.createTextNode(` ${text}`)
        );

        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
        // Speech recognition
    const SpeechRecognition =
        window.SpeechRecognition ||
        window.webkitSpeechRecognition;

    let recognition = null;
    let isListening = false;
    let isSubmitting = false;
    let stopRequested = false;
    let finalTranscript = "";
    let interimTranscript = "";
    let restartTimer = null;
    let pendingAnswer = "";

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();

        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        speakBtn.addEventListener("click", () => {
            if (interviewEnded || isSubmitting) return;

            if (isListening) {
                stopSpeakingAndSubmit();
                return;
            }

            // Stop the AI voice before the candidate starts speaking.
            stopVoiceOutput();

            finalTranscript = "";
            interimTranscript = "";
            pendingAnswer = "";
            stopRequested = false;
            isListening = true;

            speakBtn.textContent = "⏹ Stop Speaking";

            statusText.textContent =
                "Listening... Click Stop Speaking when you have finished.";

            try {
                recognition.start();
            } catch (err) {
                console.error(
                    "Could not start recognition:",
                    err
                );

                isListening = false;
                speakBtn.textContent = "🎤 Speak";

                statusText.textContent =
                    "Could not start speech recognition. Please try again.";
            }
        });

        
recognition.onresult = (event) => {
    if (interviewEnded) return;

    for (
        let i = event.resultIndex;
        i < event.results.length;
        i++
    ) {
        const transcript =
            event.results[i][0].transcript;

        if (event.results[i].isFinal) {
            finalTranscript += transcript + " ";
            interimTranscript = "";
        } else {
            interimTranscript = transcript;
        }
    }

    const liveAnswer =
        `${finalTranscript} ${interimTranscript}`.trim();

    statusText.textContent = liveAnswer
        ? `Listening: ${liveAnswer} — Click Stop Speaking when finished.`
        : "Listening to your answer...";
};


        function stopSpeakingAndSubmit() {
            if (!isListening || isSubmitting || interviewEnded) {
                return;
            }

            isListening = false;
            stopRequested = true;

            if (restartTimer) {
                clearTimeout(restartTimer);
                restartTimer = null;
            }

            pendingAnswer =
                `${finalTranscript} ${interimTranscript}`.trim();

            speakBtn.disabled = true;
            speakBtn.textContent = "Processing...";

            statusText.textContent =
                "Finishing your answer...";

            try {
                recognition.stop();
            } catch (err) {
                console.warn(
                    "Recognition was already stopped:",
                    err
                );
            }

            // Wait briefly for the browser's final speech result.
            setTimeout(() => {
                if (interviewEnded) return;

                const answer =
                    `${finalTranscript} ${interimTranscript}`.trim() ||
                    pendingAnswer;

                // Ensure the candidate spoke at least 3 words before sending to AI
if (!answer || answer.trim().split(/\s+/).length < 3) {
    speakBtn.disabled = false;
    speakBtn.textContent = "🎤 Speak";

    statusText.textContent =
        "Your answer was too short. Please speak clearly and try again.";

    return;
}

                if (isSubmitting || interviewEnded) return;

                isSubmitting = true;
                evaluationInProgress = true;

                appendMessage("User", answer);

                statusText.textContent =
                    "Evaluating your complete answer...";

                submitAnswer(answer).finally(() => {
                    isSubmitting = false;
                    evaluationInProgress = false;

                    if (interviewEnded) return;

                    if (
                        statusText.textContent !==
                        "Interview completed!"
                    ) {
                        speakBtn.disabled = false;
                        speakBtn.textContent = "🎤 Speak";
                    }
                });
            }, 350);
        }

        recognition.onerror = (event) => {
            console.error(
                "Speech recognition error:",
                event.error
            );

            if (interviewEnded) return;

            if (event.error === "no-speech" && isListening) {
                statusText.textContent =
                    "Still listening. Continue whenever you're ready.";
                return;
            }

            if (event.error === "aborted" && stopRequested) {
                return;
            }

            isListening = false;
            stopRequested = true;

            if (restartTimer) {
                clearTimeout(restartTimer);
                restartTimer = null;
            }

            speakBtn.disabled = false;
            speakBtn.textContent = "🎤 Speak";

            statusText.textContent =
                "Speech recognition error. Please click Speak and try again.";

            logEvent(
                `Speech recognition error: ${event.error}`,
                true
            );
        };

        recognition.onend = () => {
            // Some browsers stop recognition after a pause.
            // Restart it while the candidate is still answering.
            if (
                isListening &&
                !stopRequested &&
                !interviewEnded
            ) {
                restartTimer = setTimeout(() => {
                    if (
                        !isListening ||
                        stopRequested ||
                        interviewEnded
                    ) {
                        return;
                    }

                    try {
                        recognition.start();
                    } catch (err) {
                        console.warn(
                            "Recognition restart failed:",
                            err
                        );
                    }
                }, 250);
            }
        };
    } else {
        speakBtn.textContent = "Speech not supported";
        speakBtn.disabled = true;

        statusText.textContent =
            "Speech recognition is not supported in this browser. " +
            "Try Microsoft Edge or Google Chrome.";
    }

    // Submit the completed answer to the backend.
    async function submitAnswer(answerText) {
        try {
            const response = await fetch(
                `${API_BASE_URL}/evaluate-answer`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        candidate_name: candidateData.name,
                        job_role: candidateData.jobRole,
                        question: currentQuestion,
                        answer: answerText,
                        previous_answers: answerHistory
                    })
                }
            );

            if (interviewEnded) return;

            if (!response.ok) {
                throw new Error(
                    `Evaluation failed: HTTP ${response.status}`
                );
            }

            const evaluation = await response.json();

            if (interviewEnded) return;

            answerHistory.push(answerText);

            if (evaluation.similarity_flag) {
                appendMessage(
                    "System",
                    `Similarity review flag: ${
                        evaluation.similarity_reason ||
                        "This answer resembles earlier/reference text"
                    } (about ${
                        evaluation.similarity_score ?? "?"
                    }%). This is not proof of plagiarism.`
                );

                logEvent(
                    "Answer similarity flag raised for human review; not proof of plagiarism.",
                    true
                );
            }

           
const rawScore = Number(evaluation.score);

const scoreOutOf100 = Number.isFinite(rawScore)
    ? Math.max(0, Math.min(100, rawScore))
    : 0;

const scoreOutOf10 = (scoreOutOf100 / 10).toFixed(1);

const feedbackText =
    `Score: ${scoreOutOf10}/10\n` +
    `Feedback: ${evaluation.feedback || "No feedback was provided."}`;

appendMessage("AI", feedbackText);


            if (
                evaluation.next_question &&
                !evaluation.is_completed
            ) {
                currentQuestion = evaluation.next_question;

                appendMessage("AI", currentQuestion);

                statusText.textContent =
                    "Next question ready. Click Speak when you're ready.";
            } else {
                statusText.textContent = "Interview completed!";

                speakBtn.disabled = true;

                appendMessage(
                    "AI",
                    "Thank you for completing the interview. " +
                    "You may now review your performance."
                );
            }
        } catch (err) {
            if (interviewEnded) return;

            console.error("Evaluation error:", err);

            appendMessage(
                "System",
                "We couldn't evaluate your answer because the server " +
                "request failed. Please check the backend and try again."
            );

            statusText.textContent =
                "Evaluation failed. Please check the connection and retry.";
        }
    }
        // Clean up resources when the interview ends.
    function endInterview(
        reason = "Interview ended by the candidate."
    ) {
        if (interviewEnded) return;

        interviewEnded = true;
        isListening = false;
        stopRequested = true;

        // Stop pending recognition restart.
        if (restartTimer) {
            clearTimeout(restartTimer);
            restartTimer = null;
        }

        // Stop face monitoring.
        if (faceMonitorInterval) {
            clearInterval(faceMonitorInterval);
            faceMonitorInterval = null;
        }

        // Close face detector.
        if (faceLandmarker) {
            try {
                faceLandmarker.close();
            } catch (err) {
                console.warn(
                    "Could not close face detector:",
                    err
                );
            }

            faceLandmarker = null;
        }

        // Stop speech recognition.
        if (recognition) {
            try {
                recognition.abort();
            } catch (err) {
                console.warn(
                    "Could not stop recognition:",
                    err
                );
            }
        }

        // Stop AI voice output.
        stopVoiceOutput();

        // Stop camera and microphone tracks.
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => {
                track.stop();
            });

            mediaStream = null;
        }

        if (cameraPreview) {
            cameraPreview.srcObject = null;
        }

        // Stop interview timer.
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }

        statusDot.classList.remove("active");
        statusText.textContent = "Interview ended.";

        speakBtn.disabled = true;
        speakBtn.textContent = "🎤 Speak";
        endBtn.disabled = true;

        appendMessage("System", reason);
        logEvent(reason, true);
    }

    // End interview button.
    endBtn.addEventListener("click", () => {
        if (
            !confirm(
                "Are you sure you want to end the interview?"
            )
        ) {
            return;
        }

        endInterview("Interview ended by the candidate.");
    });
});
