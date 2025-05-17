document.addEventListener("DOMContentLoaded", () => {
    const tabButtons = document.querySelectorAll(".tab-btn")
    const tabContents = document.querySelectorAll(".tab-content")
    const diseaseSelect = document.getElementById("disease-select")
    const trainDiseaseSelect = document.getElementById("train-disease-select")
    const inputFieldsContainer = document.getElementById("input-fields-container")
    const predictBtn = document.getElementById("predict-btn")
    const trainBtn = document.getElementById("train-btn")
    const resultContainer = document.getElementById("result-container")
    const newPredictionBtn = document.getElementById("new-prediction-btn")
    const diseaseName = document.getElementById("disease-name")
    const predictionBadge = document.getElementById("prediction-badge")
    const probabilityBar = document.getElementById("probability-bar")
    const probabilityValue = document.getElementById("probability-value")
    const resultDetails = document.getElementById("result-details")
    const trainingResult = document.getElementById("training-result")
    const trainingDetails = document.getElementById("training-details")
    const modelComparison = document.getElementById("model-comparison")
    const addFeatureBtn = document.getElementById("add-feature-btn")
    const featuresList = document.getElementById("features-list")
    const newDiseaseName = document.getElementById("new-disease-name")
    const targetColumn = document.getElementById("target-column")
    const saveDiseaseBtn = document.getElementById("save-disease-btn")
    const supportedDiseasesList = document.getElementById("supported-diseases-list")
    const showAboutTab = document.getElementById("show-about-tab")
    const loadingOverlay = document.getElementById("loading-overlay")
    const loadingMessage = document.getElementById("loading-message")
  
    let diseases = {}
    let currentInputs = {}
  
    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        tabButtons.forEach((btn) => btn.classList.remove("active"))
        tabContents.forEach((content) => content.classList.remove("active"))
  
        button.classList.add("active")
        document.getElementById(button.dataset.tab).classList.add("active")
      })
    })
  
    showAboutTab.addEventListener("click", (e) => {
      e.preventDefault()
      tabButtons.forEach((btn) => btn.classList.remove("active"))
      tabContents.forEach((content) => content.classList.remove("active"))
  
      document.querySelector('[data-tab="about"]').classList.add("active")
      document.getElementById("about").classList.add("active")
    })
  
    async function fetchDiseases() {
      showLoading("Loading diseases...")
      try {
        const response = await fetch("/api/diseases")
        const data = await response.json()
  
        diseases = data.features
  
        populateDiseaseSelects(data.diseases)
        populateSupportedDiseasesList(data.diseases)
  
        hideLoading()
      } catch (error) {
        console.error("Error fetching diseases:", error)
        hideLoading()
        alert("Failed to load diseases. Please try again later.")
      }
    }
  
    function populateDiseaseSelects(diseasesList) {
      diseaseSelect.innerHTML = '<option value="">-- Select Disease --</option>'
      trainDiseaseSelect.innerHTML = '<option value="">-- Select Disease --</option>'
  
      diseasesList.forEach((disease) => {
        const option1 = document.createElement("option")
        option1.value = disease
        option1.textContent = capitalizeFirstLetter(disease)
        diseaseSelect.appendChild(option1)
  
        const option2 = document.createElement("option")
        option2.value = disease
        option2.textContent = capitalizeFirstLetter(disease)
        trainDiseaseSelect.appendChild(option2)
      })
    }
  
    function populateSupportedDiseasesList(diseasesList) {
      supportedDiseasesList.innerHTML = ""
  
      diseasesList.forEach((disease) => {
        const li = document.createElement("li")
        li.textContent = capitalizeFirstLetter(disease)
        supportedDiseasesList.appendChild(li)
      })
    }
  
    function generateInputFields(disease) {
      inputFieldsContainer.innerHTML = ""
      currentInputs = {}
  
      if (!diseases[disease]) return
  
      const features = diseases[disease]
      const gridDiv = document.createElement("div")
      gridDiv.className = "input-fields-grid"
  
      features.forEach((feature) => {
        const formGroup = document.createElement("div")
        formGroup.className = "form-group"
  
        const label = document.createElement("label")
        label.setAttribute("for", `input-${feature}`)
        label.textContent = formatFeatureName(feature)
  
        const input = document.createElement("input")
        input.type = "number"
        input.id = `input-${feature}`
        input.className = "form-control"
        input.placeholder = `Enter ${formatFeatureName(feature)}`
        input.dataset.feature = feature
  
        input.addEventListener("input", () => {
          if (input.value) {
            currentInputs[feature] = Number.parseFloat(input.value)
          } else {
            delete currentInputs[feature]
          }
  
          predictBtn.disabled = Object.keys(currentInputs).length !== features.length
        })
  
        formGroup.appendChild(label)
        formGroup.appendChild(input)
        gridDiv.appendChild(formGroup)
      })
  
      inputFieldsContainer.appendChild(gridDiv)
    }
  
    async function makePrediction(disease, inputData) {
      showLoading("Making prediction...")
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            disease: disease,
            input_data: inputData,
          }),
        })
  
        const result = await response.json()
  
        if (result.error) {
          throw new Error(result.error)
        }
  
        displayPredictionResult(result)
        hideLoading()
      } catch (error) {
        console.error("Error making prediction:", error)
        hideLoading()
        alert(`Failed to make prediction: ${error.message}`)
      }
    }
  
    function displayPredictionResult(result) {
      diseaseName.textContent = capitalizeFirstLetter(result.disease)
  
      if (result.prediction === 1) {
        predictionBadge.textContent = "Positive"
        predictionBadge.className = "badge positive"
      } else {
        predictionBadge.textContent = "Negative"
        predictionBadge.className = "badge negative"
      }
  
      const probabilityPercent = Math.round(result.probability * 100)
      probabilityBar.style.width = `${probabilityPercent}%`
      probabilityValue.textContent = `${probabilityPercent}%`
  
      resultDetails.innerHTML = `
              <p>The model predicts a <strong>${result.prediction === 1 ? "positive" : "negative"}</strong> result for ${capitalizeFirstLetter(result.disease)} with ${probabilityPercent}% confidence.</p>
              <p class="mt-2">This prediction is based on the input values you provided.</p>
          `
  
      resultContainer.classList.remove("hidden")
    }
  
    async function trainModel(disease) {
      showLoading("Training model...")
      try {
        const response = await fetch("/api/train", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            disease: disease,
          }),
        })
  
        const result = await response.json()
  
        if (result.error) {
          throw new Error(result.error)
        }
  
        displayTrainingResult(result)
        hideLoading()
      } catch (error) {
        console.error("Error training model:", error)
        hideLoading()
        alert(`Failed to train model: ${error.message}`)
      }
    }
  
    function displayTrainingResult(result) {
      trainingDetails.innerHTML = `
              <p><strong>Status:</strong> ${result.success ? "Success" : "Failed"}</p>
              <p><strong>Best Model:</strong> ${result.best_model}</p>
              <p><strong>Accuracy:</strong> ${(result.accuracy * 100).toFixed(2)}%</p>
              <p><strong>Message:</strong> ${result.message}</p>
          `
  
      const models = Object.keys(result.model_comparison)
      const accuracies = models.map((model) => result.model_comparison[model] * 100)
  
      modelComparison.innerHTML = ""
  
      models.forEach((model, index) => {
        const barContainer = document.createElement("div")
        barContainer.className = "model-bar-container"
  
        const modelLabel = document.createElement("div")
        modelLabel.className = "model-label"
        modelLabel.textContent = model
  
        const barWrapper = document.createElement("div")
        barWrapper.className = "bar-wrapper"
  
        const bar = document.createElement("div")
        bar.className = "model-bar"
        bar.style.width = `${accuracies[index]}%`
        bar.style.backgroundColor = model === result.best_model ? "var(--primary-color)" : "var(--secondary-color)"
  
        const accuracyLabel = document.createElement("div")
        accuracyLabel.className = "accuracy-label"
        accuracyLabel.textContent = `${accuracies[index].toFixed(2)}%`
  
        barWrapper.appendChild(bar)
        barWrapper.appendChild(accuracyLabel)
        barContainer.appendChild(modelLabel)
        barContainer.appendChild(barWrapper)
        modelComparison.appendChild(barContainer)
      })
  
      trainingResult.classList.remove("hidden")
    }
  
    async function addNewDisease(diseaseName, features, target) {
      showLoading("Adding new disease...")
      try {
        const response = await fetch("/api/add_disease", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            disease_name: diseaseName,
            features: features,
            target: target,
          }),
        })
  
        const result = await response.json()
  
        if (result.error) {
          throw new Error(result.error)
        }
  
        alert(`Disease ${diseaseName} added successfully!`)
  
        await fetchDiseases()
  
        newDiseaseName.value = ""
        targetColumn.value = "Target"
        featuresList.innerHTML = `
                  <div class="feature-item">
                      <input type="text" class="form-control feature-input" placeholder="Feature name">
                      <button class="btn icon-btn remove-feature-btn"><i class="fas fa-times"></i></button>
                  </div>
              `
  
        document.querySelector('[data-tab="prediction"]').click()
  
        hideLoading()
      } catch (error) {
        console.error("Error adding disease:", error)
        hideLoading()
        alert(`Failed to add disease: ${error.message}`)
      }
    }
  
    function addFeatureField() {
      const featureItem = document.createElement("div")
      featureItem.className = "feature-item"
  
      const input = document.createElement("input")
      input.type = "text"
      input.className = "form-control feature-input"
      input.placeholder = "Feature name"
  
      const removeBtn = document.createElement("button")
      removeBtn.className = "btn icon-btn remove-feature-btn"
      removeBtn.innerHTML = '<i class="fas fa-times"></i>'
      removeBtn.addEventListener("click", () => {
        featureItem.remove()
      })
  
      featureItem.appendChild(input)
      featureItem.appendChild(removeBtn)
      featuresList.appendChild(featureItem)
    }
  
    function showLoading(message = "Loading...") {
      loadingMessage.textContent = message
      loadingOverlay.classList.remove("hidden")
    }
  
    function hideLoading() {
      loadingOverlay.classList.add("hidden")
    }
  
    function capitalizeFirstLetter(string) {
      return string.charAt(0).toUpperCase() + string.slice(1)
    }
  
    function formatFeatureName(feature) {
      return feature
        .replace(/_/g, " ")
        .split(" ")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ")
    }
  
    diseaseSelect.addEventListener("change", () => {
      const selectedDisease = diseaseSelect.value
      if (selectedDisease) {
        generateInputFields(selectedDisease)
        predictBtn.disabled = true
      } else {
        inputFieldsContainer.innerHTML = ""
        predictBtn.disabled = true
      }
    })
  
    trainDiseaseSelect.addEventListener("change", () => {
      trainBtn.disabled = !trainDiseaseSelect.value
    })
  
    predictBtn.addEventListener("click", () => {
      const selectedDisease = diseaseSelect.value
      if (selectedDisease && Object.keys(currentInputs).length === diseases[selectedDisease].length) {
        makePrediction(selectedDisease, currentInputs)
      }
    })
  
    trainBtn.addEventListener("click", () => {
      const selectedDisease = trainDiseaseSelect.value
      if (selectedDisease) {
        trainModel(selectedDisease)
      }
    })
  
    newPredictionBtn.addEventListener("click", () => {
      resultContainer.classList.add("hidden")
      currentInputs = {}
  
      const inputs = inputFieldsContainer.querySelectorAll("input")
      inputs.forEach((input) => {
        input.value = ""
      })
  
      predictBtn.disabled = true
    })
  
    addFeatureBtn.addEventListener("click", addFeatureField)
  
    featuresList.addEventListener("click", (e) => {
      if (e.target.closest(".remove-feature-btn")) {
        const featureItem = e.target.closest(".feature-item")
        featureItem.remove()
      }
    })
  
    saveDiseaseBtn.addEventListener("click", () => {
      const disease = newDiseaseName.value.trim().toLowerCase()
      const target = targetColumn.value.trim()
  
      if (!disease) {
        alert("Please enter a disease name")
        return
      }
  
      if (!target) {
        alert("Please enter a target column name")
        return
      }
  
      const featureInputs = document.querySelectorAll(".feature-input")
      const features = []
  
      featureInputs.forEach((input) => {
        const feature = input.value.trim()
        if (feature) {
          features.push(feature.replace(/\s+/g, "_").toLowerCase())
        }
      })
  
      if (features.length === 0) {
        alert("Please add at least one feature")
        return
      }
  
      addNewDisease(disease, features, target)
    })
  
    fetchDiseases()
  })
  