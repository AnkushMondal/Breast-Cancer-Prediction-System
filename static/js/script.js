document.addEventListener('DOMContentLoaded', () => {
  const featureGroups = [
    {
      title: 'Mean Features',
      keys: [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
      ],
    },
    {
      title: 'Standard Error Features',
      keys: [
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
      ],
    },
    {
      title: 'Worst Features',
      keys: [
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst',
      ],
    },
  ];

  const formFields = document.getElementById('form-fields');
  const predictionForm = document.getElementById('prediction-form');
  const resultPill = document.getElementById('result-pill');
  const resultStatus = document.getElementById('result-status');
  const resultAction = document.getElementById('result-action');
  const resultMessage = document.getElementById('result-message');
  const historyList = document.getElementById('history-list');

  function formatLabel(key) {
    return key
      .replace(/_/g, ' ')
      .replace(/\b\w/g, char => char.toUpperCase())
      .replace(/Se/g, 'SE');
  }

  function createField(key) {
    const field = document.createElement('div');
    field.className = 'field-group';
    const id = key.replace(/\s/g, '_');

    field.innerHTML = `
      <label for="${id}">${formatLabel(key)}</label>
      <input id="${id}" data-key="${key}" type="number" step="any" placeholder="0.00" required />
    `;

    return field;
  }

  function buildForm() {
    if (!formFields) return;
    formFields.innerHTML = '';

    featureGroups.forEach(group => {
      const block = document.createElement('section');
      block.className = 'group-block';
      block.innerHTML = `<h3>${group.title}</h3><div class="input-grid"></div>`;
      const grid = block.querySelector('.input-grid');
      group.keys.forEach(key => grid.appendChild(createField(key)));
      formFields.appendChild(block);
    });
  }

  function setResult(result, message) {
    if (!resultPill || !resultStatus) return;
    const isError = result === 'Error' || result === 'Invalid input' || result === 'Connection Error';
    const isMalignant = result === 'Malignant';

    resultPill.textContent = result;
    resultPill.className = `result-pill ${isError ? 'error' : isMalignant ? 'danger' : 'success'}`;

    if (resultStatus) resultStatus.textContent = result;
    if (resultAction) {
      resultAction.textContent = isMalignant
        ? 'Consult a specialist immediately.'
        : result === 'Benign'
          ? 'Review routine follow-up with a clinician.'
          : 'Fix input values and try again.';
    }
    if (resultMessage) resultMessage.textContent = message;
  }

  if (predictionForm) {
    buildForm();

    predictionForm.addEventListener('submit', async event => {
      event.preventDefault();
      const featureData = {};
      let invalidCount = 0;

      document.querySelectorAll('[data-key]').forEach(input => {
        const key = input.getAttribute('data-key');
        const value = parseFloat(input.value);
        if (isNaN(value) || input.value.trim() === '') {
          input.classList.add('field-error');
          invalidCount += 1;
        } else {
          input.classList.remove('field-error');
          featureData[key] = value;
        }
      });

      if (invalidCount > 0 || Object.keys(featureData).length < 30) {
        setResult('Invalid input', 'Please fill all 30 fields with valid numeric values.');
        return;
      }

      setResult('Processing...', 'Analyzing clinical data...');

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(featureData),
        });

        const data = await response.json();

        if (response.ok) {
          setResult(data.result, 'Prediction complete.');
          if (historyList) {
            if (historyList.innerText.includes('No predictions')) {
              historyList.innerHTML = '';
            }
            const item = document.createElement('li');
            item.innerHTML = `<strong>${data.result}</strong> <span>${new Date().toLocaleTimeString()}</span>`;
            historyList.prepend(item);
          }
        } else {
          setResult('Error', data.error || 'Prediction failed.');
          console.error('Backend Error:', data.error);
        }
      } catch (error) {
        setResult('Connection Error', 'Unable to reach the prediction service.');
        console.error('Fetch Error:', error);
      }
    });
  }
});