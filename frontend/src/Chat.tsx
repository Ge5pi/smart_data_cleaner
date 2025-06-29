import React, { useState } from "react";
import { Upload, FileText, AlertTriangle, BarChart3, Database, CheckCircle2, Eye, Filter, Zap, TrendingUp, ArrowDownCircle } from "lucide-react";
import { useSearchParams } from "react-router-dom";

import axios from "axios";

const Chat = () => {
  const [params] = useSearchParams();
  const fileId = params.get("file_id");

  return (
    <div>
      <h2>Чат для файла: {fileId}</h2>
      {/* дальше можно отправить fileId в API */}
    </div>
  );
};

export default Chat;