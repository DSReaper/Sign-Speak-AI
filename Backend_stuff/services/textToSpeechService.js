module.exports = async function(text) 
{
  const response = await axios.post('https://texttospeech.api/', { text });
  return response.data.audioUrl;
};
