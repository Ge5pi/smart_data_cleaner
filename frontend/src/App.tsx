import { Routes, Route } from 'react-router-dom';
import { AppProvider } from './contexts/AppContext';
import Layout from './components/Layout';
import DataCleanerPage from './pages/DataCleanerPage';
import ChatPage from './pages/ChatPage';

function App() {
  return (
    <AppProvider>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DataCleanerPage />} />
          <Route path="chat" element={<ChatPage />} />
        </Route>
      </Routes>
    </AppProvider>
  );
}

export default App;