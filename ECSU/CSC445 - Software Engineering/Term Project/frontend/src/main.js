// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from "vue";
import VueRouter from "vue-router";
import App from "./App";

// router setup
import routes from "./routes/routes";

// Plugins
import GlobalComponents from "./globalComponents";
import GlobalDirectives from "./globalDirectives";

// MaterialDashboard plugin
import MaterialDashboard from "./material-dashboard";
import Chartist from "chartist";

// configure router
const router = new VueRouter({
  routes, // short for routes: routes
  linkExactActiveClass: "nav-item active"
});

Vue.prototype.$Chartist = Chartist;

Vue.use(VueRouter);
Vue.use(MaterialDashboard);
Vue.use(GlobalComponents);
Vue.use(GlobalDirectives);

import VueKeyCloak from "@dsb-norge/vue-keycloak-js";
Vue.use(VueKeyCloak, {
  config: {
    authRealm: "se_relm",
    authUrl: "http://localhost:8090/auth",
    authClientId: "se-front",
    logoutRedirectUri: ""
  },

  onReady: keycloak => {
    // eslint-disable-next-line
    console.log(`Keycloak response: ${keycloak}`)
    new Vue({
      el: "#app",
      render: h => h(App),
      router,
      data: {
        Chartist: Chartist
      }
    });
  }
});
