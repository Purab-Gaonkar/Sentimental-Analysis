import React, { useState } from 'react';
import axios from 'axios';
import { Pie, Bar } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, BarElement, CategoryScale, LinearScale } from 'chart.js';
import './App.css';  // Import the CSS for styling

// Register chart.js elements
ChartJS.register(ArcElement, Tooltip, Legend, BarElement, CategoryScale, LinearScale);

const App = () => {
  const [url, setUrl] = useState('');
  const [sentimentResult, setSentimentResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Function to handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError('');
    setSentimentResult(null);

    try {
      // Send URL to Flask backend for analysis
      const response = await axios.post('http://127.0.0.1:5000/analyze', { url });

      setSentimentResult(response.data);
    } catch (err) {
      setError('Error occurred while analyzing the video. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Prepare data for charts
  const getChartData = () => {
    if (!sentimentResult) return {};

    const sentimentSummary = sentimentResult.sentiment_summary;
    return {
      pie: {
        labels: ['Positive', 'Negative', 'Neutral'],
        datasets: [{
          data: [
            sentimentSummary.Positive,
            sentimentSummary.Negative,
            sentimentSummary.Neutral
          ],
          backgroundColor: ['#36A2EB', '#FF6384', '#FFCE56'],
        }],
      },
      bar: {
        labels: ['Positive', 'Negative', 'Neutral'],
        datasets: [{
          label: 'Sentiment Count',
          data: [
            sentimentSummary.Positive,
            sentimentSummary.Negative,
            sentimentSummary.Neutral
          ],
          backgroundColor: '#4CAF50', // Color for bars
        }],
      },
    };
  };

  return (
    <div className="App">
      <h1>YouTube Video Sentiment Analysis</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Enter YouTube Video URL:
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="e.g., https://www.youtube.com/watch?v=SAb4zRyxrD4"
            required
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {sentimentResult && (
        <div>
          <h2>Sentiment Analysis for Video: {sentimentResult.videoId}</h2>
          <h3>Overall Sentiment: {sentimentResult.overall_sentiment}</h3>

          <div>
            <h4>Sentiment Summary:</h4>
            <ul>
              <li>Positive: {sentimentResult.sentiment_summary.Positive}</li>
              <li>Negative: {sentimentResult.sentiment_summary.Negative}</li>
              <li>Neutral: {sentimentResult.sentiment_summary.Neutral}</li>
            </ul>
          </div>

          {/* Pie Chart for Sentiment Distribution */}
          <h4>Sentiment Distribution</h4>
          <Pie data={getChartData().pie} />

          {/* Bar Chart for Sentiment Count */}
          <h4>Sentiment Count</h4>
          <Bar data={getChartData().bar} options={{
            responsive: true,
            scales: {
              y: {
                beginAtZero: true
              }
            }
          }} />

          <h4>Analyzed Comments:</h4>
          <ul>
            {sentimentResult.analyzed_comments.map((comment, index) => (
              <li key={index}>
                <strong>Sentiment:</strong> {comment.sentiment} ({comment.probability})
                <p>{comment.comment}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default App;
