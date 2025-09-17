module.exports = async function(videoPath) 
{
  const response = await axios.post('http://localhost:8001', { videoPath });
  return response.data.text;
};
