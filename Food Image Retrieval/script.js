console.log("SCRIPT LOADED");
document.addEventListener("DOMContentLoaded", function () {

    const imageInput = document.getElementById("imageInput");
    const preview = document.getElementById("preview");
    const searchBtn = document.getElementById("searchBtn");
    const maeBtn = document.getElementById("maeBtn");

    // IMAGE PREVIEW
    imageInput.addEventListener("change", function () {
        console.log("image selected");
        const file = this.files[0];
        if (!file) return;

        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.style.width = "100%";
        img.style.height = "100%";
        img.style.objectFit = "cover";

        preview.innerHTML = "";
        preview.appendChild(img);
    });

    // SEARCH BUTTON
    searchBtn.addEventListener("click", uploadImage);
    maeBtn.addEventListener("click", runMAE);

});

// IMAGE UPLOAD + SEARCH

async function uploadImage() {

    const input = document.getElementById("imageInput");
    const statusDiv = document.getElementById("status");
    const resultsDiv = document.getElementById("results");
    const maeResults = document.getElementById("maeResults");
    const similarSection = document.getElementById("similarSection");
    const maeSection = document.getElementById("maeSection");

    // Hide combined title
    document.getElementById("combinedTitle").style.display = "none";

    // Show section title for similar images
    document.getElementById("similarTitle").style.display = "block";
    document.getElementById("maeTitle").style.display = "none";

    // CLEAR MAE RESULTS when doing search
    maeResults.innerHTML = "";
    resultsDiv.innerHTML = "";

    if (!input.files.length) {
        alert("Please select an image!");
        return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);

    statusDiv.innerHTML = "Processing...";

    try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();

        resultsDiv.innerHTML = "";

        data.results.forEach(item => {

            const card = document.createElement("div");
            card.className = "result-card";

            const img = document.createElement("img");
            img.src = item.image;

            const score = document.createElement("div");
            score.className = "score";
            score.innerText = `${item.label} | ${item.score}`;

            card.appendChild(img);
            card.appendChild(score);

            resultsDiv.appendChild(card);

        });

        statusDiv.innerHTML = `✅ Found ${data.results.length} similar images`;

    } catch (err) {
        console.error("UPLOAD ERROR:", err);
        statusDiv.innerHTML = "❌ Error while uploading image!";
    }
}


// MAE Reconstruction
async function runMAE() {
    const similarSection = document.getElementById("similarSection");
    const maeSection  = document.getElementById("maeSection");
    const input = document.getElementById("imageInput");
    const maeResults = document.getElementById("maeResults");
    const resultsDiv = document.getElementById("results");
    const statusDiv = document.getElementById("status");

    // Hide combined title
    document.getElementById("combinedTitle").style.display = "none";

    // Show section title for MAE
    document.getElementById("similarTitle").style.display = "none";
    document.getElementById("maeTitle").style.display = "block";

    // CLEAR previous results
    resultsDiv.innerHTML = "";
    maeResults.innerHTML = "";

    if (!input.files.length) {
        alert("Please select an image!");
        return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);

    statusDiv.innerHTML = "Running MAE reconstruction...";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Server error");

        const data = await response.json();

        const images = [
            {title: "Original", img: data.original_image},
            {title: "Reconstructed", img: data.reconstructed_image},
            {title: "MAE Output", img: data.mae_output_image}
        ];

        images.forEach(item => {
            const card = document.createElement("div");
            card.className = "result-card";

            const img = document.createElement("img");
            img.src = `data:image/png;base64,${item.img}`;

            const label = document.createElement("div");
            label.className = "score";
            label.innerText = item.title;

            card.appendChild(img);
            card.appendChild(label);

            maeResults.appendChild(card);
        });

        statusDiv.innerHTML = "✅ MAE reconstruction completed";

    } catch (err) {
        console.error("MAE ERROR:", err);
        statusDiv.innerHTML = "❌ Error running MAE";
    }
}