async function rateJoke() {
  const joke = document.getElementById("jokeInput").value;
  const model = document.getElementById("modelSelect").value;
  const loadingIndicator = document.getElementById("loadingIndicator");

  if (!joke.trim()) {
    alert("Por favor, ingresa un chiste antes de calificarlo.");
    return;
  }

  loadingIndicator.classList.remove("hidden"); // Mostrar "Cargando..."

  try {
    const response = await fetch("/rate-joke", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ joke, model }),
    });

    const data = await response.json();
    console.log(data);

    if (!response.ok) {
      console.error("Error:", data.error || "Error desconocido");
      alert("Error al procesar el chiste: " + (data.error || "Intenta con otro modelo."));
      return;
    }

    const humorLabel = document.getElementById("humorLabel");
    const label = data.is_funny ? "✅ Es un chiste" : "❌ No es un chiste";
    const confidence = Math.round(data.confidence * 100) + "%";
    humorLabel.textContent = `${label} (confianza: ${confidence})`;

    humorLabel.style.backgroundColor = data.is_funny ? "#d4edda" : "#f8d7da";
    humorLabel.style.color = "#333";
    humorLabel.style.border = "1px solid #ccc";
    humorLabel.style.padding = "10px";
    humorLabel.style.borderRadius = "8px";
    humorLabel.style.fontWeight = "bold";
    humorLabel.style.textAlign = "center";
    humorLabel.style.maxWidth = "437px";
    humorLabel.style.margin = "15px auto 10px";

    if (data.is_funny) {
      document.getElementById("humorLabelContanier").classList.remove("hidden");
      document.getElementById("circleHumorContainer").classList.remove("hidden");
      document.getElementById("score").textContent = data.score.toFixed(1);

      const emojiEl = document.getElementById("emoji");
      const score = data.score;
      let emoji = "🤣";
      let feedback = "";
      let context = "";

      if (score === 5) {
        emoji = "🤣";
        feedback = "¡Muy gracioso! Tal vez tengas futuro en la comedia.";
        context = "Este chiste está en el 15% superior.";
      } else if (score === 4) {
        emoji = "😄";
        feedback = "¡Nada mal! Tienes un buen sentido del humor.";
        context = "Este chiste está en el 40% superior.";
      } else if (score === 3) {
        emoji = "😐";
        feedback = "Está bien... tal vez le falte un poco más de chispa.";
        context = "Este chiste es promedio.";
      } else if (score === 2) {
        emoji = "😕";
        feedback = "Hmm... hemos visto mejores.";
        context = "Este chiste está en el 40% inferior.";
      } else {
        emoji = "😩";
        feedback = "Uy... tal vez no dejes tu trabajo actual.";
        context = "Este chiste está en el 10% inferior.";
      }

      emojiEl.textContent = emoji;
      document.getElementById("feedbackText").textContent = feedback;
      document.getElementById("contextText").textContent = context;

      const radius = 50;
      const circumference = 2 * Math.PI * radius;
      const offset = circumference - (score / 5) * circumference;
      const bar = document.querySelector(".circle-bar");
      bar.style.strokeDashoffset = offset;
    } else {
      document.getElementById("humorLabelContanier").classList.remove("hidden");
      document.getElementById("circleHumorContainer").classList.add("hidden");
    }

  } catch (error) {
    console.error("Error inesperado:", error);
    alert("Ocurrió un error inesperado. Intenta nuevamente.");
  } finally {
    loadingIndicator.classList.add("hidden"); // Ocultar "Cargando..."
  }
}

function openModal() {
  document.getElementById("creatorsModal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("creatorsModal").classList.add("hidden");
}

window.onload = function () {
  fetch("/models")
    .then((response) => response.json())
    .then((models) => {
      const select = document.getElementById("modelSelect");
      select.innerHTML = "";
      models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        select.appendChild(option);
      });
      if (models.length > 0) {
        select.value = models[0];
      }
    })
    .catch((error) => {
      console.error("Error al cargar los modelos:", error);
    });
};
