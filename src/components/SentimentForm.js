// src/components/SentimentForm.js

import React, { useState } from "react";
import axios from "axios";

const SentimentForm = ({ setSentiment, setProbability }) => {
    const [url, setUrl] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();

        setLoading(true);
        setError(null);

        try {
            // Send the URL to your backend for sentiment analysis
            const response = await axios.post("http://localhost:5000/analyze", {
                text: url, // send the URL or process comments from this URL
            });
            setSentiment(response.data.sentiment);
            setProbability(response.data.probability);
        } catch (err) {
            setError("Failed to analyze sentiment. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h1>Sentiment Analysis for YouTube Comments</h1>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    placeholder="Enter YouTube video URL"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                />
                <button type="submit" disabled={loading}>
                    {loading ? "Analyzing..." : "Analyze"}
                </button>
            </form>

            {error && <p>{error}</p>}
        </div>
    );
};

export default SentimentForm;
