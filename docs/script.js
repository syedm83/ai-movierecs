const availableMoods = [
  'drama', 'comedy', 'sci-fi', 'horror', 'thriller', 
  'action', 'crime', 'romance', 'adventure', 'animation',
  'family', 'musical', 'historical', 'inspirational', 'dark comedy',
  'psychological', 'fantasy', 'mystery', 'biography', 'war'
];

let model;
let movieData = [];
let selectedMoods = [];

async function loadEmbeddings() {
  const res = await fetch("movie_embeddings.json");
  movieData = await res.json();

  movieData.forEach(movie => {
    movie.tagsArray = movie.tags
      .toLowerCase()
      .split(/,|\•/)
      .map(tag => tag.trim());
  });
}

function createMoodButtons() {
  const container = document.getElementById('moodContainer');
  
  availableMoods.forEach(mood => {
    const btn = document.createElement('button');
    btn.className = 'mood-btn';
    btn.textContent = mood;
    btn.addEventListener('click', () => toggleMoodSelection(mood, btn));
    container.appendChild(btn);
  });
}

function toggleMoodSelection(mood, btnElement) {
  const index = selectedMoods.indexOf(mood);
  
  if (index === -1) {
    if (selectedMoods.length < 3) {
      selectedMoods.push(mood);
      btnElement.classList.add('selected');
    }
  } else {
    selectedMoods.splice(index, 1);
    btnElement.classList.remove('selected');
  }
  
  updateMoodButtonsState();
  document.getElementById('recommendBtn').disabled = selectedMoods.length === 0;
}

function updateMoodButtonsState() {
  const buttons = document.querySelectorAll('.mood-btn');
  buttons.forEach(btn => {
    btn.classList.toggle('limit-reached', 
      selectedMoods.length >= 3 && !selectedMoods.includes(btn.textContent)
    );
  });
}

async function recommendMovies() {
  if (selectedMoods.length === 0) return;

  const query = selectedMoods.join(' ');
  const inputEmbedding = await model.embed([query]);
  const inputArray = inputEmbedding.arraySync()[0];

  const recommended = movieData.map(movie => {
    const similarity = cosineSimilarity(inputArray, movie.embedding);

    const matchedTags = selectedMoods.filter(mood =>
      movie.tagsArray.includes(mood.toLowerCase())
    );
    const tagMatchBoost = matchedTags.length * 0.07;

    return {
      title: movie.title,
      tags: movie.tags,
      similarity: similarity + tagMatchBoost,
      baseSimilarity: similarity,
      tagMatchBoost
    };
  })
  .filter(movie => movie.similarity > 0.1) 
  .sort((a, b) => b.similarity - a.similarity)
  .slice(0, 5);

  displayResults(recommended);
}




function cosineSimilarity(a, b) {
  const dim = 512;
  const processedA = a.length >= dim ? a.slice(0, dim) : [...a, ...new Array(dim - a.length).fill(0)];
  const processedB = b.length >= dim ? b.slice(0, dim) : [...b, ...new Array(dim - b.length).fill(0)];
  
  const tfA = tf.tensor1d(processedA);
  const tfB = tf.tensor1d(processedB);
  const dotProduct = tf.dot(tfA, tfB).dataSync()[0];
  const normA = tf.norm(tfA).dataSync()[0];
  const normB = tf.norm(tfB).dataSync()[0];
  return dotProduct / (normA * normB);
}

function displayResults(movies) {
  const resultDiv = document.getElementById('result');
  
  if (!movies || movies.length === 0) {
    resultDiv.innerHTML = `<p>No strong matches. Try combining moods like "crime + drama" or "sci-fi + action"</p>`;
    return;
  }

  resultDiv.innerHTML = '<h2>🎬 Recommended Movies</h2>';
  
  movies.forEach(movie => {
    resultDiv.innerHTML += `
      <div class="movie-card">
        <h3>${movie.title}</h3>
        <p>Tags: ${movie.tags.replace(/, /g, ' • ')}</p>
      </div>
    `;
  });
}

window.onload = async () => {
  try {
    await tf.ready();
    model = await use.load();
    
    await loadEmbeddings();
    
    createMoodButtons();
    
    document.getElementById('recommendBtn').addEventListener('click', recommendMovies);
    
    console.log("App initialized successfully!");
  } catch (error) {
    console.error("Initialization failed:", error);
    alert("Failed to load app. Check console for errors.");
  }
};