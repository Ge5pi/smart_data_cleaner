import { useState } from 'react'
import axios from 'axios'

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<Record<string, any>[]>([])

  const handleUpload = async () => {
    if (!file) return
    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await axios.post("http://localhost:5643/upload/", formData)
      setPreview(res.data.preview)
    } catch (err) {
      console.error("Ошибка при загрузке файла", err)
    }
  }

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold mb-4">Smart Data Cleaner</h1>
      <input type="file" accept=".csv" onChange={e => setFile(e.target.files?.[0] || null)} />
      <button onClick={handleUpload} className="ml-2 px-4 py-1 bg-blue-500 text-white">
        Загрузить
      </button>

      {preview.length > 0 && (
        <table className="mt-6 border">
          <thead>
            <tr>
              {Object.keys(preview[0]).map((col) => (
                <th key={col} className="border px-2">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.map((row, i) => (
              <tr key={i}>
                {Object.values(row).map((val, j) => (
                  <td key={j} className="border px-2">{val}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default App
