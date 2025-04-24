import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';
import { FiSearch, FiBarChart2, FiPieChart, FiActivity, FiDownload, FiRefreshCw, FiInfo } from 'react-icons/fi';
import { format } from 'date-fns';

// Custom theme colors
const THEME = {
  primary: '#4F46E5',
  primaryLight: '#818CF8',
  secondary: '#10B981',
  neutral: '#6B7280',
  danger: '#EF4444',
  warning: '#F59E0B',
  background: '#F9FAFB',
  card: '#FFFFFF',
  text: '#1F2937',
  textLight: '#6B7280',
};

const App = () => {
  const [keywords, setKeywords] = useState('climate,environment');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [searchHistory, setSearchHistory] = useState([]);

  // Handle keyword input change
  const handleKeywordChange = (e) => {
    setKeywords(e.target.value);
  };

  // Fetch results from the API
  const fetchResults = async () => {
    // Add to search history
    if (keywords.trim()) {
      setSearchHistory(prev => {
        const newHistory = [{ keywords, timestamp: new Date().toISOString() }, ...prev];
        return newHistory.slice(0, 5); // Keep only last 5 searches
      });
    }

    setLoading(true);
    setError(null);
    try {
      const response = await axios.get('https://fb3b0ih3xh.execute-api.us-east-1.amazonaws.com/NewsApiSentinement', {
        params: { keywords },
      });

      console.log('API Response:', response.data);
      setResults(response.data);
      const { csvLink } = response.data;

      if (!csvLink) {
        throw new Error('CSV link not found in response');
      }

      // Fetch CSV file and parse it
      const csvResponse = await axios.get(csvLink);
      Papa.parse(csvResponse.data, {
        header: true,
        skipEmptyLines: true,
        complete: (result) => {
          console.log('Parsed CSV:', result);

          // Clean and format data
          const cleanedData = result.data.map(item => ({
            headline: item.Headline || '',
            url: item.URL || '',
            source: item.Source || '',
            publishedAt: item.PublishedAt || '',
            timestamp: item.Timestamp || '',
            compound: parseFloat(item.Compound) || 0,
            positive: parseFloat(item.Positive) || 0,
            neutral: parseFloat(item.Neutral) || 0,
            negative: parseFloat(item.Negative) || 0
          }));

          setData(cleanedData);
          // Automatically show dashboard after loading
          setActiveTab('dashboard');
        },
        error: (err) => {
          console.error('CSV Parse Error:', err);
          setError('Error parsing CSV data');
        },
      });
    } catch (err) {
      console.error('Fetch Error:', err);
      setError(`Error fetching data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Sort headlines by sentiment score (most positive to most negative)
  const getSortedHeadlines = () => {
    if (!data) return [];
    return [...data].sort((a, b) => b.compound - a.compound);
  };

  // Calculate average sentiment
  const getAverageSentiment = () => {
    if (!data || data.length === 0) return 0;
    const sum = data.reduce((acc, item) => acc + item.compound, 0);
    return (sum / data.length).toFixed(2);
  };

  // Get sentiment category count
  const getSentimentCounts = () => {
    if (!data) return { positive: 0, neutral: 0, negative: 0 };

    return data.reduce((counts, item) => {
      if (item.compound > 0.05) counts.positive++;
      else if (item.compound < -0.05) counts.negative++;
      else counts.neutral++;
      return counts;
    }, { positive: 0, neutral: 0, negative: 0 });
  };

  // Get top keywords from headlines
  const getTopKeywords = () => {
    if (!data) return [];

    const keywordCounts = {};
    const stopWords = ['the', 'and', 'of', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'as'];

    data.forEach(item => {
      const words = item.headline.toLowerCase().split(/\s+/);
      words.forEach(word => {
        // Remove punctuation and filter stop words
        const cleanWord = word.replace(/[^a-z0-9]/g, '');
        if (cleanWord && cleanWord.length > 3 && !stopWords.includes(cleanWord)) {
          keywordCounts[cleanWord] = (keywordCounts[cleanWord] || 0) + 1;
        }
      });
    });

    // Convert to array and sort
    return Object.entries(keywordCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([keyword, count]) => ({ keyword, count }));
  };

  // Generate sentiment chart data
  const sentimentChartData = {
    labels: data ? data.map(item => {
      // Truncate long headlines for chart labels
      const headline = item.headline;
      return headline.length > 30 ? headline.substring(0, 30) + '...' : headline;
    }) : [],
    datasets: [
      {
        label: 'Sentiment Score',
        data: data ? data.map(item => item.compound) : [],
        backgroundColor: data ? data.map(item =>
          item.compound > 0.05 ? `${THEME.secondary}99` :
            item.compound < -0.05 ? `${THEME.danger}99` :
              `${THEME.primaryLight}99`
        ) : [],
        borderColor: data ? data.map(item =>
          item.compound > 0.05 ? THEME.secondary :
            item.compound < -0.05 ? THEME.danger :
              THEME.primaryLight
        ) : [],
        borderWidth: 1
      }
    ]
  };

  // Sentiment distribution chart data
  const sentimentDistributionData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        label: 'Number of Headlines',
        data: (() => {
          const counts = getSentimentCounts();
          return [counts.positive, counts.neutral, counts.negative];
        })(),
        backgroundColor: [
          `${THEME.secondary}99`,
          `${THEME.primaryLight}99`,
          `${THEME.danger}99`
        ],
        borderColor: [
          THEME.secondary,
          THEME.primaryLight,
          THEME.danger
        ],
        borderWidth: 1
      }
    ]
  };

  // Sentiment composition radar chart data
  const sentimentCompositionData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        label: 'Average Sentiment Composition',
        data: !data ? [0, 0, 0] : [
          data.reduce((sum, item) => sum + item.positive, 0) / data.length,
          data.reduce((sum, item) => sum + item.neutral, 0) / data.length,
          data.reduce((sum, item) => sum + item.negative, 0) / data.length
        ],
        backgroundColor: `${THEME.primary}33`,
        borderColor: THEME.primary,
        borderWidth: 2,
        pointBackgroundColor: THEME.primary,
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: THEME.primary
      }
    ]
  };

  const keywordFrequencyData = {
    labels: getTopKeywords().map(item => item.keyword),
    datasets: [
      {
        label: 'Frequency',
        data: getTopKeywords().map(item => item.count),
        backgroundColor: `${THEME.primary}99`,
        borderColor: THEME.primary,
        borderWidth: 1
      }
    ]
  };

  const chartOptions = {
    scales: {
      y: {
        beginAtZero: false,
        suggestedMin: -1,
        suggestedMax: 1,
        grid: {
          color: '#e2e8f0'
        }
      },
      x: {
        grid: {
          color: '#e2e8f0'
        }
      }
    },
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: {
            family: "'Inter', sans-serif",
            size: 12
          },
          color: THEME.text
        }
      }
    }
  };

  // Get sentiment color based on score
  const getSentimentColor = (score) => {
    if (score > 0.05) return 'text-green-600';
    if (score < -0.05) return 'text-red-600';
    return 'text-blue-600';
  };

  // Get sentiment background color based on score
  const getSentimentBgColor = (score) => {
    if (score > 0.05) return 'bg-green-100';
    if (score < -0.05) return 'bg-red-100';
    return 'bg-blue-100';
  };

  // Get sentiment label based on score
  const getSentimentLabel = (score) => {
    if (score > 0.05) return 'Positive';
    if (score < -0.05) return 'Negative';
    return 'Neutral';
  };

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    try {
      return format(new Date(timestamp), 'MMM d, yyyy • h:mm a');
    } catch (e) {
      return timestamp;
    }
  };

  // Download results as CSV
  const downloadCsv = () => {
    if (!data) return;

    const csvData = Papa.unparse(data);
    const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `sentiment-analysis-${format(new Date(), 'yyyy-MM-dd-HHmm')}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      {/* Navbar */}
      <nav className="bg-indigo-600 text-white px-6 py-4 shadow-lg">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between">
          <div className="flex items-center mb-4 md:mb-0">
            <FiActivity className="text-2xl mr-2" />
            <span className="text-xl font-bold">NewsLens</span>
            <span className="ml-2 bg-indigo-800 text-xs px-2 py-1 rounded">Sentiment Analysis Platform</span>
          </div>
          <div className="flex space-x-4">
            <button onClick={() => setActiveTab('dashboard')} className={`py-1 px-3 rounded transition-colors ${activeTab === 'dashboard' ? 'bg-white text-indigo-600' : 'hover:bg-indigo-500'}`}>
              Dashboard
            </button>
            <button onClick={() => setActiveTab('data')} className={`py-1 px-3 rounded transition-colors ${activeTab === 'data' ? 'bg-white text-indigo-600' : 'hover:bg-indigo-500'}`}>
              Data Table
            </button>
            <button onClick={() => setActiveTab('about')} className={`py-1 px-3 rounded transition-colors ${activeTab === 'about' ? 'bg-white text-indigo-600' : 'hover:bg-indigo-500'}`}>
              About
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search Section */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-8">
          <div className="flex flex-col md:flex-row md:items-end gap-4">
            <div className="flex-grow">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Enter Keywords for News Analysis
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <FiSearch className="text-gray-400" />
                </div>
                <input
                  type="text"
                  value={keywords}
                  onChange={handleKeywordChange}
                  className="w-full pl-10 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="climate,environment,technology"
                />
              </div>
              <p className="mt-1 text-xs text-gray-500">Separate multiple keywords with commas</p>
            </div>
            <button
              onClick={fetchResults}
              className="px-6 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 flex items-center"
            >
              <FiBarChart2 className="mr-2" />
              Analyze Headlines
            </button>
          </div>

          {/* Search history */}
          {searchHistory.length > 0 && (
            <div className="mt-4">
              <p className="text-xs text-gray-500 mb-2">Recent searches:</p>
              <div className="flex flex-wrap gap-2">
                {searchHistory.map((item, index) => (
                  <button
                    key={index}
                    onClick={() => setKeywords(item.keywords)}
                    className="inline-flex items-center px-2 py-1 rounded-full bg-gray-100 hover:bg-gray-200 text-xs text-gray-600"
                  >
                    {item.keywords}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {loading && (
          <div className="flex justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500 mx-auto"></div>
              <p className="mt-4 text-gray-600">Analyzing news headlines...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8 rounded-md">
            <div className="flex">
              <div className="flex-shrink-0">
                <FiInfo className="text-red-500" />
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && results && data && (
          <div className="space-y-8">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-xl shadow-md p-6 transition-transform hover:scale-105">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-700">Keywords</h3>
                  <div className="p-2 bg-indigo-100 rounded-lg">
                    <FiSearch className="text-indigo-600" />
                  </div>
                </div>
                <p className="text-2xl font-bold">{keywords}</p>
                <p className="text-sm text-gray-500 mt-2">{data.length} headlines analyzed</p>
              </div>

              <div className="bg-white rounded-xl shadow-md p-6 transition-transform hover:scale-105">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-700">Average Sentiment</h3>
                  <div className={`p-2 ${getSentimentBgColor(getAverageSentiment())} rounded-lg`}>
                    <FiActivity className={getSentimentColor(getAverageSentiment())} />
                  </div>
                </div>
                <p className={`text-2xl font-bold ${getSentimentColor(getAverageSentiment())}`}>
                  {getAverageSentiment()}
                </p>
                <p className="text-sm text-gray-500 mt-2">{getSentimentLabel(getAverageSentiment())} overall sentiment</p>
              </div>

              <div className="bg-white rounded-xl shadow-md p-6 transition-transform hover:scale-105">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-700">Sentiment Distribution</h3>
                  <div className="p-2 bg-green-100 rounded-lg">
                    <FiPieChart className="text-green-600" />
                  </div>
                </div>
                <div className="flex justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Positive</p>
                    <p className="text-xl font-bold text-green-600">{getSentimentCounts().positive}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Neutral</p>
                    <p className="text-xl font-bold text-blue-600">{getSentimentCounts().neutral}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Negative</p>
                    <p className="text-xl font-bold text-red-600">{getSentimentCounts().negative}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-xl shadow-md p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-800">Sentiment Distribution</h2>
                  <div className="bg-gray-100 rounded-md p-1">
                    <FiPieChart className="text-gray-600" />
                  </div>
                </div>
                <div className="h-80">
                  <Doughnut
                    data={sentimentDistributionData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'bottom',
                          labels: {
                            font: {
                              family: "'Inter', sans-serif",
                              size: 12
                            }
                          }
                        }
                      }
                    }}
                  />
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-md p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-800">Sentiment Composition</h2>
                  <div className="bg-gray-100 rounded-md p-1">
                    <FiActivity className="text-gray-600" />
                  </div>
                </div>
                <div className="h-80">
                  <Radar
                    data={sentimentCompositionData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        r: {
                          angleLines: {
                            display: true
                          },
                          suggestedMin: 0,
                          suggestedMax: 1
                        }
                      }
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-xl shadow-md p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-800">Headlines Sentiment Scores</h2>
                  <div className="bg-gray-100 rounded-md p-1">
                    <FiBarChart2 className="text-gray-600" />
                  </div>
                </div>
                <div className="h-80">
                  <Bar data={sentimentChartData} options={chartOptions} />
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-md p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-xl font-semibold text-gray-800">Top Keywords Frequency</h2>
                  <div className="bg-gray-100 rounded-md p-1">
                    <FiBarChart2 className="text-gray-600" />
                  </div>
                </div>
                <div className="h-80">
                  <Bar
                    data={keywordFrequencyData}
                    options={{
                      indexAxis: 'y',
                      ...chartOptions
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Recent Headlines */}
            <div className="bg-white rounded-xl shadow-md p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold text-gray-800">Top 5 Headlines</h2>
                <button
                  onClick={() => setActiveTab('data')}
                  className="text-sm text-indigo-600 hover:text-indigo-800"
                >
                  View all headlines →
                </button>
              </div>
              <div className="space-y-4">
                {getSortedHeadlines().slice(0, 5).map((item, index) => (
                  <div key={index} className="border-b border-gray-100 pb-4 last:border-0">
                    <div className="flex justify-between">
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-base font-medium text-gray-900 hover:text-indigo-600"
                      >
                        {item.headline}
                      </a>
                      <div className={`ml-2 px-3 py-1 rounded-full text-xs font-medium ${getSentimentBgColor(item.compound)} ${getSentimentColor(item.compound)}`}>
                        {item.compound.toFixed(2)}
                      </div>
                    </div>
                    <div className="flex justify-between mt-2">
                      <span className="text-sm text-gray-500">{item.source}</span>
                      <span className="text-sm text-gray-500">{formatTimestamp(item.timestamp)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Export Section */}
            <div className="bg-white rounded-xl shadow-md p-6">
              <div className="flex flex-col sm:flex-row justify-between items-center">
                <div>
                  <h2 className="text-xl font-semibold text-gray-800">Export Results</h2>
                  <p className="text-gray-500 mt-1">Download the full analysis results</p>
                </div>
                <div className="mt-4 sm:mt-0 flex space-x-3">
                  <a
                    href={results.csvLink}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
                  >
                    <FiRefreshCw className="mr-2" />
                    View Raw Data
                  </a>
                  <button
                    onClick={downloadCsv}
                    className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
                  >
                    <FiDownload className="mr-2" />
                    Download CSV
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Data Table Tab */}
        {activeTab === 'data' && data && (
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold text-gray-800">Headlines Analysis</h2>
              <button
                onClick={downloadCsv}
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
              >
                <FiDownload className="mr-2" />
                Download CSV
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Headline</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Published</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sentiment</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Positive</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Neutral</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Negative</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {getSortedHeadlines().map((item, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-normal">
                        <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                          {item.headline}
                        </a>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.source}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatTimestamp(item.publishedAt)}
                      </td>
                      <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${getSentimentColor(item.compound)}`}>
                        <span className={`px-2 py-1 rounded-full text-xs ${getSentimentBgColor(item.compound)}`}>
                          {item.compound.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.positive.toFixed(2)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.neutral.toFixed(2)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.negative.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* About Tab */}
        {activeTab === 'about' && (
          <div className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">About NewsLens</h2>
            <div className="prose max-w-none">
              <p className="mb-4">
                NewsLens is a powerful sentiment analysis platform designed to help you understand the emotional tone behind news headlines. Using advanced natural language processing techniques, it analyzes news headlines to determine whether they convey positive, neutral, or negative sentiments.
              </p>
              <h3 className="text-xl font-semibold text-gray-700 mt-6 mb-3">How It Works</h3>
              <p className="mb-4">
                The platform fetches news headlines from a variety of sources based on your specified keywords. Each headline is then processed through a sentiment analysis algorithm that evaluates the emotional tone of the text. The analysis breaks down the sentiment into three components:
              </p>

              <ul className="list-disc pl-5 mb-4">
                <li className="mb-2"><strong className="text-indigo-600">Compound Score:</strong> An overall sentiment score between -1 (extremely negative) and +1 (extremely positive).</li>
                <li className="mb-2"><strong className="text-green-600">Positive:</strong> The degree of positive emotion expressed in the headline.</li>
                <li className="mb-2"><strong className="text-blue-600">Neutral:</strong> The degree of neutral or objective content in the headline.</li>
                <li className="mb-2"><strong className="text-red-600">Negative:</strong> The degree of negative emotion expressed in the headline.</li>
              </ul>

              <h3 className="text-xl font-semibold text-gray-700 mt-6 mb-3">Features</h3>
              <ul className="list-disc pl-5 mb-4">
                <li className="mb-2"><strong>Keyword Analysis:</strong> Search for specific topics using keywords.</li>
                <li className="mb-2"><strong>Visual Dashboard:</strong> Interactive charts and graphs to visualize sentiment data.</li>
                <li className="mb-2"><strong>Data Export:</strong> Download the full analysis results in CSV format.</li>
                <li className="mb-2"><strong>Recent Headlines:</strong> View and sort news headlines by sentiment score.</li>
                <li className="mb-2"><strong>Keyword Frequency:</strong> Identify common terms used in headlines about your topic.</li>
              </ul>

              <h3 className="text-xl font-semibold text-gray-700 mt-6 mb-3">Use Cases</h3>
              <ul className="list-disc pl-5 mb-4">
                <li className="mb-2"><strong>Media Analysis:</strong> Understand how different topics are covered in the news.</li>
                <li className="mb-2"><strong>Brand Monitoring:</strong> Track public perception of your brand or industry.</li>
                <li className="mb-2"><strong>Research:</strong> Gather data on public sentiment for academic or business research.</li>
                <li className="mb-2"><strong>Content Strategy:</strong> Inform your content creation based on current news sentiment.</li>
              </ul>

              <h3 className="text-xl font-semibold text-gray-700 mt-6 mb-3">Technology</h3>
              <p className="mb-4">
                NewsLens is built using a modern tech stack including:
              </p>
              <ul className="list-disc pl-5 mb-4">
                <li className="mb-2"><strong>Front-end:</strong> React.js with a responsive design</li>
                <li className="mb-2"><strong>Data Visualization:</strong> Chart.js for interactive charts</li>
                <li className="mb-2"><strong>Sentiment Analysis:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner)</li>
                <li className="mb-2"><strong>API:</strong> RESTful API for fetching news and analyzing sentiment</li>
                <li className="mb-2"><strong>Data Processing:</strong> Real-time processing of news headlines</li>
              </ul>

              <h3 className="text-xl font-semibold text-gray-700 mt-6 mb-3">Privacy & Data</h3>
              <p className="mb-4">
                NewsLens respects your privacy. We do not store your search queries beyond your current session, and analysis results are temporary. All data is processed securely and not shared with third parties.
              </p>

              <div className="bg-indigo-50 p-4 rounded-lg mt-8">
                <h4 className="font-semibold text-indigo-800 mb-2">Feedback & Support</h4>
                <p className="text-indigo-700">
                  We're constantly improving NewsLens based on user feedback. If you have suggestions or encounter any issues, please contact our support team.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-500 text-sm pb-8">
          <p>NewsLens © {new Date().getFullYear()} | Sentiment Analysis Platform</p>
          <p className="mt-1">Made with ❤️ for news and data enthusiasts</p>
        </footer>
      </div>
    </div>
  );
};

export default App;