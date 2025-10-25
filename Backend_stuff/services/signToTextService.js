module.exports = async function(videoPath) 
{
  const response = await axios.post('https://ssai.belgiumcampus.ac.za', { videoPath });
  return response.data.text;
};
