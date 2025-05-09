const form = document.getElementById('forecast-form');
const resultsDiv = document.getElementById('forecast-data');
const loadingDiv = document.getElementById('loading');
const errorMessageDiv = document.getElementById('error-message');
const forecastTypeSelect = document.getElementById('forecast_type');
const dayOfWeekGroup = document.getElementById('day-of-week-group');
const dayOfWeekSelect = document.getElementById('day_of_week');

const API_BASE_URL = 'http://127.0.0.1:8000';

forecastTypeSelect.addEventListener('change', () => {
    if (forecastTypeSelect.value === '1week') {
        dayOfWeekGroup.style.display = 'block';
    } else {
        dayOfWeekGroup.style.display = 'none';
        dayOfWeekSelect.value = '';
    }
});

form.addEventListener('submit', async (event) => {
    event.preventDefault();

    resultsDiv.innerHTML = '';
    errorMessageDiv.style.display = 'none';
    errorMessageDiv.textContent = '';
    loadingDiv.style.display = 'block';

    const city = document.getElementById('city').value;
    const model_name = document.getElementById('model_name').value;
    const forecast_type = forecastTypeSelect.value;
    const day_of_week = dayOfWeekSelect.value;

    const params = new URLSearchParams({
        city: city,
        model_name: model_name,
        forecast_type: forecast_type,
    });

    if (forecast_type === '1week' && day_of_week !== '') {
        params.append('day_of_week', day_of_week);
    }

    const apiUrl = `${API_BASE_URL}/predict?${params.toString()}`;

    console.log(`Fetching: ${apiUrl}`);

    try {
        const response = await fetch(apiUrl);

        if (!response.ok) {
            let errorMsg = `Error: ${response.status} ${response.statusText}`;
            try {
                const errorData = await response.json();
                errorMsg = `Error ${response.status}: ${errorData.detail || response.statusText}`;
            } catch (e) {
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();

        if (data && data.length > 0) {
            renderResults(data);
        } else {
             resultsDiv.innerHTML = '<p>No forecast data available for the selected criteria.</p>';
        }

    } catch (error) {
        console.error('Fetch error:', error);
        errorMessageDiv.textContent = `Failed to fetch forecast: ${error.message}`;
        errorMessageDiv.style.display = 'block';
    } finally {
        loadingDiv.style.display = 'none';
    }
});

function renderResults(data) {
    if (!data || data.length === 0) {
        resultsDiv.innerHTML = '<p>No forecast data received.</p>';
        return;
    }

    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');
    const headerRow = document.createElement('tr');

    const headers = Object.keys(data[0]);
    headers.forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    data.forEach(rowData => {
        const row = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = rowData[header];
            row.appendChild(td);
        });
        tbody.appendChild(row);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
    resultsDiv.appendChild(table);
}