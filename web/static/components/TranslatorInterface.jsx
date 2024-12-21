const TranslatorInterface = () => {
    const [models, setModels] = React.useState([]);
    const [currentModel, setCurrentModel] = React.useState(null);
    const [sourceText, setSourceText] = React.useState('');
    const [translation, setTranslation] = React.useState('');
    const [isLoading, setIsLoading] = React.useState(false);
    const [error, setError] = React.useState('');
  
    React.useEffect(() => {
      fetchModels();
    }, []);
  
    const fetchModels = async () => {
      try {
        const response = await fetch('/models');
        const data = await response.json();
        setModels(data.models);
        if (data.models.length > 0 && !currentModel) {
          setCurrentModel(data.models[0]);
        }
      } catch (err) {
        setError('Failed to load models');
      }
    };
  
    const handleModelSelect = (model) => {
      setCurrentModel(model);
      setError('');
      setTranslation('');
    };
  
    const handleTranslate = async () => {
      if (!sourceText.trim()) {
        setError('Please enter text to translate');
        return;
      }
      
      if (!currentModel) {
        setError('Please select a model');
        return;
      }
  
      setIsLoading(true);
      setError('');
      setTranslation('');
  
      try {
        const response = await fetch('/translate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: sourceText,
            model_id: currentModel.id
          }),
        });
        
        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || 'Translation failed');
        }
        
        const data = await response.json();
        setTranslation(data.translation);
      } catch (err) {
        setError(`Translation failed: ${err.message}`);
      } finally {
        setIsLoading(false);
      }
    };
  
    return (
      <div className="max-w-3xl mx-auto p-6">
        <h1 className="text-2xl font-bold mb-6">English to Russian Translator</h1>
        
        <div className="flex flex-wrap gap-3 mb-6">
          {models.map((model) => (
            <button
              key={model.id}
              onClick={() => handleModelSelect(model)}
              className={`px-4 py-2 border-2 rounded-md transition-colors ${
                currentModel && currentModel.id === model.id
                  ? 'bg-blue-500 text-white border-blue-500'
                  : 'bg-white text-gray-700 border-gray-300 hover:border-blue-300'
              }`}
            >
              {model.name}
            </button>
          ))}
        </div>
        
        {currentModel && (
          <div className="text-sm text-gray-600 mb-4">
            {currentModel.description} (max {currentModel.max_length} words)
          </div>
        )}
        
        <div className="mb-4">
          <label className="block text-gray-700 mb-2">English text:</label>
          <textarea
            value={sourceText}
            onChange={(e) => setSourceText(e.target.value)}
            className="w-full h-36 p-3 border rounded-md focus:ring-2 focus:ring-blue-300 focus:border-blue-300 outline-none"
            placeholder="Enter English text here..."
          />
        </div>
        
        <button
          onClick={handleTranslate}
          disabled={isLoading}
          className="px-6 py-3 bg-green-600 text-white rounded-md text-lg mb-4 hover:bg-green-700 disabled:bg-green-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 min-w-[120px]"
        >
          {isLoading ? (
            React.createElement(React.Fragment, null,
              React.createElement("svg", {
                className: "animate-spin h-5 w-5",
                viewBox: "0 0 24 24"
              },
                React.createElement("circle", {
                  className: "opacity-25",
                  cx: "12",
                  cy: "12",
                  r: "10",
                  stroke: "currentColor",
                  strokeWidth: "4",
                  fill: "none"
                }),
                React.createElement("path", {
                  className: "opacity-75",
                  fill: "currentColor",
                  d: "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                })
              ),
              React.createElement("span", null, "Translating...")
            )
          ) : (
            'Translate'
          )}
        </button>
        
        {error && (
          <div className="p-3 mb-4 bg-red-50 text-red-700 rounded-md border border-red-200">
            {error}
          </div>
        )}
        
        <div>
          <label className="block text-gray-700 mb-2">Russian translation:</label>
          <textarea
            value={translation}
            readOnly
            className="w-full h-36 p-3 border rounded-md bg-gray-50"
          />
        </div>
      </div>
    );
}