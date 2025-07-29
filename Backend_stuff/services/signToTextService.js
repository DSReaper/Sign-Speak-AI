module.exports = async function(videoPath) 
{
  const response = await axios.post('http://localhost:8000', { videoPath });
  return response.data.text;
};
