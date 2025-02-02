// src/components/SentimentResult.js

import React from "react";

const SentimentResult = ({ sentiment, probability }) => {
    return (
        <div>
            <h2>Sentiment: {sentiment}</h2>
            <p>Probability: {probability}</p>
        </div>
    );
};

export default SentimentResult;
