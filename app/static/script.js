async function rateJoke() {
    const joke = document.getElementById("jokeInput").value;
    const model = document.getElementById("modelSelect").value;
  
    const response = await fetch("/rate-joke", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ joke, model }),
    });
  
    const data = await response.json();
  
    // Mostrar resultado
    document.getElementById("resultContainer").classList.remove("hidden");
    document.getElementById("score").textContent = data.score.toFixed(1);
  
    // Emoji
    const emojiEl = document.getElementById("emoji");
    const score = data.score;
    let emoji = "🤣";
    let feedback = "";
    let context = "";
  
    if (score >= 8) {
      emoji = "🤣";
      feedback = "Pretty funny! You might have a future in comedy.";
      context = "This joke ranks in the top 15% of all submissions";
    } else if (score >= 6) {
      emoji = "😄";
      feedback = "Not bad! You’ve got some comedic flair.";
      context = "This joke ranks in the top 40% of all submissions";
    } else if (score >= 4) {
      emoji = "😐";
      feedback = "It’s okay... maybe it needs a little more punch.";
      context = "This joke ranks around average.";
    } else if (score >= 2) {
      emoji = "😕";
      feedback = "Hmm... we’ve seen better.";
      context = "This joke ranks in the bottom 40%.";
    } else {
      emoji = "😩";
      feedback = "Yikes. Maybe keep your day job.";
      context = "This joke ranks in the bottom 10%.";
    }
  
    emojiEl.textContent = emoji;
    document.getElementById("feedbackText").textContent = feedback;
    document.getElementById("contextText").textContent = context;
  
    // Círculo animado
    const radius = 50;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 10) * circumference;
    const bar = document.querySelector(".circle-bar");
    bar.style.strokeDashoffset = offset;
  }  

  function openModal() {
    document.getElementById("creatorsModal").classList.remove("hidden");
  }
  
  function closeModal() {
    document.getElementById("creatorsModal").classList.add("hidden");
  }
  
  