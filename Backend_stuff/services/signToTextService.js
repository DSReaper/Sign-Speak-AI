module.exports = async function(videoPath) 
{
  const response = await axios.post('http://localhost:8000/model', { videoPath });
  return response.data.text;
};
